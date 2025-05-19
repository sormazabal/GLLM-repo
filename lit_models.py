import lightning.pytorch as pl
import torchmetrics
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import torch

from base import BaseModel
from utils.runner.metric import c_index
from calflops import calculate_flops
CUDA_LAUNCH_BLOCKING=1

class LitFullModel(pl.LightningModule):
    def __init__(self, models: dict[str, BaseModel], optimizers: dict[str, torch.optim.Optimizer], config: dict):
        super().__init__()
        for model_name, m in models.items():
            config[f'model.{model_name}'] = m.__structure__()
        self.save_hyperparameters(config)
        self.feat_ext = models['feat_ext']
        self.classifier = models['clf']
        self.optimizers_dict = optimizers
        self.step_results = []    
        self.bootstrap_status = {
    
                'index': [],
                'project_id': [],
                'output': [],
                'target': [],
                'survival_time': [],
                'vital_status': []
            }     
        self.bootstrap_index = 0                                         # Slow but clean.
        # Disable automatic optimization for manual backward if there are multiple optimizers.
        if 'all' not in self.optimizers_dict:
            self.automatic_optimization = False
          # Initialize FLOPS counters
        # Initialize placeholders for FLOPs, MACs, and Params
        self.flops_feat_ext = None
        self.macs_feat_ext = None
        self.params_feat_ext = None
        self.flops_classifier = None
        self.macs_classifier = None
        self.params_classifier = None

    def configure_optimizers(self):
        if 'all' in self.optimizers_dict:
            return self.optimizers_dict['all']
        return [self.optimizers_dict['feat_ext'], self.optimizers_dict['clf']]
    
    def calculate_model_flops(self, model, input_tensors):
        if model is not None:
            flops, macs, params = calculate_flops(model=model, 
                                                  input_shape=input_tensors.shape, 
                                                  output_as_string=True,
                                                  output_precision=4)
            return flops, macs, params
        return None, None, None

    def training_step(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch

        if isinstance(self.optimizers(), list):
            self.optimizers()[0].zero_grad()
            self.optimizers()[1].zero_grad()

         # Calculate FLOPs for feature extractor
        # if self.flops_feat_ext is None:
        #     input_tensors = (genomic, clinical, project_id)
        #     self.flops_feat_ext, self.macs_feat_ext, self.params_feat_ext = self.calculate_model_flops(self.feat_ext, genomic)
        #     self.logger.info(f"Feature Extractor FLOPs: {self.flops_feat_ext}   MACs: {self.macs_feat_ext}   Params: {self.params_feat_ext}")



        embedding = self.feat_ext(genomic, clinical, project_id)
         # Calculate FLOPs for classifier
        # if self.flops_classifier is None:
        #     self.flops_classifier, self.macs_classifier, self.params_classifier = self.calculate_model_flops(self.classifier, embedding)
        #     self.logger.info(f"Classifier FLOPs: {self.flops_classifier}   MACs: {self.macs_classifier}   Params: {self.params_classifier}")



        if isinstance(project_id, list):
            project_id = torch.tensor(project_id, device=embedding.device)
        y = self.classifier(embedding, project_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)

        if isinstance(self.optimizers(), list):
            self.manual_backward(loss)
            self.optimizers()[0].step()
            self.optimizers()[1].step()
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def _shared_eval(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        y = self.classifier(self.feat_ext(genomic, clinical, project_id), project_id)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)
        #print(f"y shape: {y.shape}, overall_survival shape: {overall_survival.shape}")
        

        self.step_results.append({
            'output': y.detach().cpu(),
            'label': overall_survival.detach().cpu().type(torch.int64),
            'survival_time': survival_time.detach().cpu(),
            'vital_status': vital_status.detach().cpu(),
            #'project_id': project_id.detach().cpu(),
            'project_id': project_id if isinstance(project_id, list) else project_id.detach().cpu(),
        })
        return loss

    def _shared_epoch_end(self) -> None:
        outputs = torch.cat([result['output'] for result in self.step_results])
        
        if torch.isnan(outputs).any():
            print(f"Any NaNs in model outputs: {torch.isnan(outputs).any()}")
            print(f"Model outputs (before mask): {outputs}")

        outputs = torch.functional.F.sigmoid(outputs)                           # AUC and PRC will not be affected.
        labels = torch.cat([result['label'] for result in self.step_results])
        survival_time = torch.cat([result['survival_time'] for result in self.step_results])
        vital_status = torch.cat([result['vital_status'] for result in self.step_results])
        project_id = torch.cat([result['project_id'] for result in self.step_results])
        
        self.bootstrap_status['index'].extend([self.bootstrap_index] * len(project_id.detach().cpu()))
        self.bootstrap_status['project_id'].extend(project_id.detach().cpu())
        self.bootstrap_status['output'].extend(outputs.detach().cpu())
        self.bootstrap_status['target'].extend(labels.detach().cpu())
        self.bootstrap_status['survival_time'].extend(survival_time.detach().cpu())
        self.bootstrap_status['vital_status'].extend(vital_status.detach().cpu())
        self.bootstrap_index += 1

        for i in torch.unique(project_id):
            mask = project_id == i
            roc = torchmetrics.functional.auroc(outputs[mask], labels[mask], 'binary')
            prc = torchmetrics.functional.average_precision(outputs[mask], labels[mask], 'binary')
            assert not torch.isnan(outputs[mask]).any(), "NaNs detected in outputs[mask]"
            assert not torch.isnan(survival_time[mask]).any(), "NaNs detected in survival_time[mask]"
            assert not torch.isnan(vital_status[mask]).any(), "NaNs detected in vital_status[mask]"

            cindex = c_index(outputs[mask], survival_time[mask], vital_status[mask])
            self.log(f'PRC_{i}', prc, on_epoch=True, on_step=False)
            self.log(f'AUC_{i}', roc, on_epoch=True, on_step=False)            
            self.log(f'C-Index_{i}', cindex, on_epoch=True, on_step=False)
        self.step_results.clear()

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx)
        self.log('loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        y = self.classifier(self.feat_ext(genomic, clinical, project_id), project_id)
        return y.detach().cpu().numpy()
    
    def return_bootstrap_status(self):
        return self.bootstrap_status


class LitBayesianClassifier(LitFullModel):
    def __init__(self, lit_models: LitFullModel, config: dict):
        super(LitFullModel, self).__init__()
        self.save_hyperparameters(config)
        self.feat_ext = lit_models.feat_ext
        self.classifier = lit_models.classifier
        dnn_to_bnn(self.classifier, config['bayesian']['prior'])
        self.optimizer_hparams = config['bayesian']['optimizer']
        self.step_results = []                                                  # Slow but clean.

    def configure_optimizers(self):
        for param in self.feat_ext.parameters():
            param.requires_grad = False
        if 'AdamW' in self.optimizer_hparams:
            opt = torch.optim.AdamW(self.classifier.parameters(),
                                    lr=self.optimizer_hparams['AdamW']['lr'],
                                    weight_decay=self.optimizer_hparams['AdamW']['weight_decay'])
        elif 'SGD' in self.optimizer_hparams:
            opt = torch.optim.SGD(self.classifier.parameters(),
                                  lr=self.optimizer_hparams['SGD']['lr'],
                                  momentum=self.optimizer_hparams['SGD']['momentum'])
        return opt

    def training_step(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        embedding = self.feat_ext(genomic, clinical, project_id)
        y = self.classifier(embedding, project_id).squeeze()
        kl_loss = get_kl_loss(self.classifier) / embedding.shape[0]
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(y, overall_survival)
        loss = bce_loss + kl_loss
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_kl_loss', kl_loss, on_epoch=True, on_step=False)
        self.log('train_bce_loss', bce_loss, on_epoch=True, on_step=False)
        return loss

    def _shared_eval(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        output_mc = []
        for _ in range(self.hparams['bayesian']['mc_samples']):
            output_mc.append(self.classifier(self.feat_ext(genomic, clinical, project_id), project_id))
        output = torch.stack(output_mc).mean(dim=0).squeeze()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, overall_survival)
        self.step_results.append({
            'output': output.detach().cpu(),
            'label': overall_survival.detach().cpu().type(torch.int64),
            'survival_time': survival_time.detach().cpu(),
            'vital_status': vital_status.detach().cpu(),
            'project_id': project_id.detach().cpu(),
        })
        return loss


class LitCancerType(LitFullModel):
    def __init__(self, lit_models: LitFullModel, optimizers_dict: dict[str, torch.optim.Optimizer]):
        super(LitFullModel, self).__init__()
        self.feat_ext = lit_models.feat_ext
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(8, 3),
            torch.nn.BatchNorm1d(3),
        )
        self.lr = optimizers_dict['clf'].param_groups[0]['lr']
        self.momentum = optimizers_dict['clf'].param_groups[0]['momentum']
        self.step_outputs = []
        self.step_labels = []

    def configure_optimizers(self):
        for param in self.feat_ext.parameters():
            param.requires_grad = False
        return torch.optim.SGD(self.classifier.parameters(), lr=self.lr, momentum=self.momentum)

    def training_step(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        embedding = self.feat_ext(genomic, clinical, project_id)
        # Takes the genomic embedding only. The first 8 dimensions are the genomic embedding.
        y = self.classifier(embedding[:, 8:])
        loss = torch.nn.functional.cross_entropy(y, project_id)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def _shared_eval(self, batch, batch_idx):
        (genomic, clinical, index, project_id), (overall_survival, survival_time, vital_status) = batch
        # Takes the genomic embedding only. The first 8 dimensions are the genomic embedding.
        y = self.classifier(self.feat_ext(genomic, clinical, project_id)[:, 8:])
        loss = torch.nn.functional.cross_entropy(y, project_id)
        self.step_outputs.append(y.detach().cpu())
        self.step_labels.append(project_id.detach().cpu())
        return loss

    def _shared_epoch_end(self):
        outputs = torch.cat(self.step_outputs)
        labels = torch.cat(self.step_labels)
        acc = torchmetrics.functional.accuracy(outputs, labels, 'multiclass', num_classes=3)
        self.log('ACC', acc, on_epoch=True, prog_bar=True)
        self.step_outputs.clear()
        self.step_labels.clear()


class LitFineTuning(LitFullModel):
    def __init__(self, models: dict[str, BaseModel], optimizers: dict[str, torch.optim.Optimizer], config: dict,
                 pretrained_genomic_models: LitFullModel, pretrained_clinical_models: LitFullModel):
        super().__init__(models, optimizers, config)
        self._load_pretrained(pretrained_genomic_models, pretrained_clinical_models)

        # Reset optimizer. 'params' must be reset.
        # HACK: Only use one optimizer for the whole model. Use requires_grad = False to freeze the model.
        del self.optimizers_dict['all'].param_groups[0]['params']
        self.optimizers_dict['all'] = getattr(torch.optim, optimizers['all'].__class__.__name__)(
            [param for param in self.parameters()],
            **optimizers['all'].param_groups[0]
        )

    def _load_pretrained(self, pretrained_genomic_models: LitFullModel, pretrained_clinical_models: LitFullModel):
        assert hasattr(pretrained_genomic_models, 'feat_ext')
        assert hasattr(pretrained_genomic_models.feat_ext, 'genomic_feature_extractor')
        self.feat_ext.genomic_feature_extractor = pretrained_genomic_models.feat_ext.genomic_feature_extractor
        for param in self.feat_ext.genomic_feature_extractor.parameters():
            param.requires_grad = False

        assert hasattr(pretrained_clinical_models, 'feat_ext')
        assert hasattr(pretrained_clinical_models.feat_ext, 'clinical_feature_extractor')
        self.feat_ext.clinical_feature_extractor = pretrained_clinical_models.feat_ext.clinical_feature_extractor
        for param in self.feat_ext.clinical_feature_extractor.parameters():
            param.requires_grad = False
