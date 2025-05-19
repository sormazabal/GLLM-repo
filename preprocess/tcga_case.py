import copy
from utils.logger import get_logger
from pathlib import Path
import pandas as pd
from utils.api.tcga_api import get_metadata_from_case, get_filters_result_from_file, download_file, download_files
import lxml.etree as LET


class TCGA_Case(object):
    '''
    TCGA Case
    '''
    def __init__(self, case_id, directory, case_metadata=None, case_file_paths=None, genomic_type='tpm'):
        '''
        Initialize the TCGA Case instance with parameters.

        Needed parameters
        :param case_id: Specify the case id.
        :param directory: Specify the directory for the downloaded files.

        Optional parameters
        :param case_metadata: The metadata of the case.
        :param case_file_paths: The related file paths of the case.
        :param genomic_type: Specify the wanted genomic type that you want to use.
        '''
        # Case ID
        self.case_id = case_id
        if case_metadata:
            self.case_metadata = case_metadata
        else:
            self.case_metadata = self._get_case_metadata(case_id=self.case_id)

        # Logger
        self.logger = get_logger(name='preprocess.tcga_case')

        # Directory for download files
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)

        if case_file_paths:
            self.case_file_paths = case_file_paths
        else:
            self.case_file_paths = self._get_case_file_paths(case_id=self.case_id, directory=self.directory)

            # Download files
            self._download_files(
                case_metadata=self.case_metadata,
                case_file_paths=self.case_file_paths,
                extract_directory=self.directory
            )

        self.case_file_paths = sorted(self.case_file_paths)

        # Genomic type
        self.genomic_type = genomic_type

        # Testing

    def _get_case_metadata(self, case_id):
        '''
        Get the metadata according to case id from TCGA API.

        :param case_id: Specify the case id.
        '''
        kwargs = {}

        # Expand
        kwargs['expand'] = [
            'annotations',
            'diagnoses',
            'diagnoses.annotations',
            'diagnoses.pathology_details',
            'diagnoses.treatments',
            'demographic',
            'exposures',
            'family_histories',
            'files',
            'follow_ups',
            'follow_ups.molecular_tests',
            'samples',
            'samples.annotations',
            'samples.portions',
            'samples.portions.analytes',
            'samples.portions.analytes.aliquots',
            'samples.portions.analytes.aliquots.annotations',
            'samples.portions.analytes.aliquots.center',
            'samples.portions.analytes.annotations',
            'samples.portions.annotations',
            'samples.portions.center',
            'samples.portions.slides',
            'samples.portions.slides.annotations'
        ]

        return get_metadata_from_case(case_id=case_id, **kwargs)

    def _get_case_file_paths(self, case_id, directory):
        '''
        Get the related file paths according to case id from TCGA API.

        :param case_id: Specify the case id.
        :param directory: The parent directory of the related files.
        '''
        kwargs = {}
        kwargs['size'] = get_metadata_from_case(case_id=case_id, fields=['summary.file_count'])['summary']['file_count']

        # Filter
        kwargs['filters'] = {
            'and': [
                {'=': {'cases.case_id': case_id}},
                {'=': {'access': 'open'}},
                {'or': [
                    {'and': [
                        {'=': {'data_type': 'Clinical Supplement'}},
                        {'=': {'data_format': 'BCR XML'}}
                    ]},
                    {'and': [
                        {'=': {'data_type': 'Gene Expression Quantification'}},
                        {'=': {'experimental_strategy': 'RNA-Seq'}},
                        {'=': {'data_format': 'TSV'}}
                    ]}
                ]}
            ]
        }

        # Fields
        kwargs['fields'] = ['cases.case_id', 'file_name', 'file_id']

        return [directory.joinpath(file_metadata['file_name'])
                for file_metadata in get_filters_result_from_file(**kwargs)]

    def _download_files(self, case_metadata, case_file_paths, extract_directory):
        '''
        Download the related files according to case id.

        :param case_metadata: The metadata of the case.
        :param case_file_paths: The related file paths of the case.
        :param extract_directory: Specify the directory for downloading and extracting the files.
        '''
        # Check if the data exists
        total_file_names = [file_path.name for file_path in case_file_paths]
        exist_file_names = [file_path.name for file_path in extract_directory.rglob('*') if file_path.is_file()]
        download_file_names = list(set(total_file_names) - set(exist_file_names))
        download_file_ids = []

        for case_file_metadata in case_metadata['files']:
            if case_file_metadata['file_name'] in download_file_names:
                download_file_ids.append(case_file_metadata['file_id'])

        # Download files from tcga api
        if len(download_file_ids) == 1:
            self.logger.info('Downloading 1 file for {}...'.format(self.case_id))
            download_file(file_id=download_file_ids[0], extract_directory=str(extract_directory))
        elif len(download_file_ids) > 1:
            self.logger.info('Downloading {} files for {}...'.format(len(download_file_ids), self.case_id))
            download_files(file_ids=download_file_ids, extract_directory=str(extract_directory))

    def _get_genomic_data(self, case_file_paths):
        '''
        Read all the genomic data from related files.

        :param case_file_paths: A list of the related file paths.
        '''
        rna_seq = []

        for case_file_path in case_file_paths:
            if case_file_path.suffix not in ['.tsv']:
                self.logger.debug(f'Abandon {case_file_path} while handling genomic data')
                continue

            df_data = pd.read_csv(case_file_path, sep='\t', skiprows=1, index_col='gene_name')

            # Filter out unwanted columns
            wanted_columns = ['unstranded', 'stranded_first', 'stranded_second',
                              'tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded']
            df_data = df_data[df_data.gene_type == 'protein_coding'][wanted_columns]

            rna_seq.append(df_data)

        # TODO: Need to decide which method is going to be used
        rna_seq = pd.concat(rna_seq, axis=1).groupby(level=0, axis=1).mean()
        rna_seq = rna_seq.groupby(level='gene_name').sum()

        return rna_seq

    def _get_clinical_data(self, case_id, case_metadata):
        '''
        Read all the clinical data from the case's metadata.

        :param case_id: Specify the case id.
        :param case_metadata: The metadata of the case.
        '''
        if len(case_metadata['diagnoses']) == 1:
            clinical = copy.deepcopy(case_metadata['diagnoses'][0])
        else:
            raise ValueError(f'More than one diagnosis for {case_id}')

        clinical.update(case_metadata['demographic'])

        if len(case_metadata['exposures']) == 1:
            clinical.update(case_metadata['exposures'][0])
        else:
            raise ValueError(f'More than one exposure for {case_id}')

        clinical.update({'disease_type': case_metadata['disease_type']})
        clinical.update({'primary_site': case_metadata['primary_site']})
        clinical.update({'samples': case_metadata['samples']})

        return pd.DataFrame.from_dict(clinical, orient='index', columns=[case_id])

    def _get_vital_status_data(self, case_id, case_metadata):
        '''
        Read the vital status data from the case's metadata.

        :param case_id: Specify the case id.
        :param case_metadata: The metadata of the case.
        '''
        vital_status = case_metadata['demographic']['vital_status']
        if vital_status == 'Alive':
            vital_status = 0
        elif vital_status == 'Dead':
            vital_status = 1
        else:
            raise ValueError('Vital status must be alive or dead')

        return pd.DataFrame([vital_status], columns=[case_id], index=['vital_status'])

    def _get_overall_survival_data(self, case_id, case_metadata, year):
        '''
        Calculate the overall survival data from the case's metadata.

        :param case_id: Specify the case id.
        :param case_metadata: The metadata of the case.
        '''
        vital_status = case_metadata['demographic']['vital_status']
        if vital_status == 'Alive':
            overall_survival = 0
        elif vital_status == 'Dead':

            days_to_death = case_metadata['demographic']['days_to_death']
            days_to_diagnosis = case_metadata['diagnoses'][0]['days_to_diagnosis']

            if days_to_death - days_to_diagnosis > year * 365:
                overall_survival = 0
            else:
                overall_survival = 1
        else:
            raise ValueError('Vital status must be alive or dead')

        return pd.DataFrame([overall_survival], columns=[case_id], index=['overall_survival'])

    def _get_disease_specific_survival_data(self, case_id, case_metadata, case_file_paths, year):
        '''
        Calculate the disease specific survival data from the case's metadata and clinical file.

        :param case_id: Specify the case id.
        :param case_metadata: The metadata of the case.
        '''
        clinical_file_paths = []
        for case_file_path in case_file_paths:
            if case_file_path.suffix in ['.xml']:
                clinical_file_paths.append(case_file_path)

        if len(clinical_file_paths) > 1:
            raise ValueError(f'More than one clinical xml for {case_id}')
        elif len(clinical_file_paths) == 0:
            raise ValueError(f'No clinical xml for {case_id}')

        clinical_file_path = clinical_file_paths[0]

        clinical_parser = LET.XMLParser(
            ns_clean=True,
            remove_blank_text=True
        )
        clinical_tree = LET.parse(clinical_file_path, clinical_parser)
        clinical_root = clinical_tree.getroot()

        # Remove all empty nodes
        while len(clinical_tree.xpath('.//*[not(*) and not(text())]')):
            for element in clinical_tree.xpath('.//*[not(*) and not(text())]'):
                element.getparent().remove(element)

        person_neoplasm_cancer_status = clinical_tree.xpath(
            './/*[local-name()="patient"]/clin_shared:person_neoplasm_cancer_status',
            namespaces=clinical_root.nsmap
        )
        if len(person_neoplasm_cancer_status) > 1:
            raise ValueError(f'More than one person neoplasm cancer status for {case_id}')
        elif len(person_neoplasm_cancer_status) == 0:
            self.logger.debug(f'No person neoplasm cancer status for {case_id}, using last follow up instead')
            person_neoplasm_cancer_status = None
        else:
            person_neoplasm_cancer_status = person_neoplasm_cancer_status[0]

        follow_ups = clinical_tree.xpath(
            './/*[local-name()="follow_ups"]//clin_shared:person_neoplasm_cancer_status',
            namespaces=clinical_root.nsmap
        )
        if len(follow_ups):
            last_follow_up = follow_ups[0]
            for follow_up in follow_ups:
                if LET.QName(follow_up.getparent().tag).namespace > LET.QName(last_follow_up.getparent().tag).namespace:
                    last_follow_up = follow_up

            if person_neoplasm_cancer_status is not None:
                if last_follow_up.text != person_neoplasm_cancer_status.text:
                    self.logger.debug(
                        f'Person neoplasm cancer status is not match with the last follow up for {case_id},'
                        'using last follow up instead'
                    )
                    person_neoplasm_cancer_status = last_follow_up
            else:
                person_neoplasm_cancer_status = last_follow_up
        else:
            if person_neoplasm_cancer_status is None:
                self.logger.debug(f'No person neoplasm cancer status for {case_id}')
                disease_specific_survival = -1
                return pd.DataFrame([disease_specific_survival], columns=[case_id], index=['disease_specific_survival'])

        person_neoplasm_cancer_status = person_neoplasm_cancer_status.text

        vital_status = case_metadata['demographic']['vital_status']
        if vital_status == 'Alive':
            disease_specific_survival = -1
        elif vital_status == 'Dead':
            if person_neoplasm_cancer_status == 'WITH TUMOR':
                days_to_death = case_metadata['demographic']['days_to_death']
                days_to_diagnosis = case_metadata['diagnoses'][0]['days_to_diagnosis']

                if days_to_death - days_to_diagnosis > year * 365:
                    disease_specific_survival = 0
                else:
                    disease_specific_survival = 1
            elif person_neoplasm_cancer_status == 'TUMOR FREE':
                disease_specific_survival = -1
            else:
                raise ValueError(f'Person neoplasm cancer status {person_neoplasm_cancer_status}')
        else:
            raise ValueError('Vital status must be alive or dead')

        return pd.DataFrame([disease_specific_survival], columns=[case_id], index=['disease_specific_survival'])

    def _get_survival_time_data(self, case_id, case_metadata, year):
        '''
        Calculate the survival time data from the case's metadata.

        :param case_id: Specify the case id.
        :param case_metadata: The metadata of the case.
        '''
        vital_status = case_metadata['demographic']['vital_status']
        if vital_status == 'Alive':
            survival_time = year * 365
        elif vital_status == 'Dead':

            days_to_death = case_metadata['demographic']['days_to_death']
            days_to_diagnosis = case_metadata['diagnoses'][0]['days_to_diagnosis']

            survival_time = days_to_death - days_to_diagnosis
        else:
            raise ValueError('Vital status must be alive or dead')

        return pd.DataFrame([survival_time], columns=[case_id], index=['survival_time'])

    def _get_primary_site_data(self, case_id, case_metadata):
        '''
        Read the primary site data from the case's metadata.

        :param case_id: Specify the case id.
        :param case_metadata: The metadata of the case.
        '''
        primary_site = case_metadata['primary_site']

        return pd.DataFrame([primary_site], columns=[case_id], index=['primary_site'], dtype='category')

    @property
    def genomic(self):
        '''
        Return the genomic data.
        '''
        rna_seq = self._get_genomic_data(case_file_paths=self.case_file_paths)

        if self.genomic_type == 'fpkm':
            return rna_seq['fpkm_unstranded'].to_frame(name=self.case_id)
        elif self.genomic_type == 'fpkm-uq':
            return rna_seq['fpkm_uq_unstranded'].to_frame(name=self.case_id)
        elif self.genomic_type == 'tpm':
            return rna_seq['tpm_unstranded'].to_frame(name=self.case_id)
        elif self.genomic_type == 'unstranded':
            return rna_seq['unstranded'].to_frame(name=self.case_id)
        elif self.genomic_type == 'stranded-first':
            return rna_seq['stranded_first'].to_frame(name=self.case_id)
        elif self.genomic_type == 'stranded-second':
            return rna_seq['stranded_second'].to_frame(name=self.case_id)
        else:
            raise KeyError('Wrong genomic type')

    @property
    def clinical(self):
        '''
        Return the clinical data.
        '''
        return self._get_clinical_data(case_id=self.case_id, case_metadata=self.case_metadata)

    @property
    def vital_status(self):
        '''
        Return the vital status data.
        '''
        return self._get_vital_status_data(case_id=self.case_id, case_metadata=self.case_metadata)

    @property
    def overall_survival(self, year=5):
        '''
        Return the overall survival data.
        '''
        return self._get_overall_survival_data(case_id=self.case_id, case_metadata=self.case_metadata, year=year)

    @property
    def disease_specific_survival(self, year=5):
        '''
        Return the disease specific survival data.
        '''
        return self._get_disease_specific_survival_data(case_id=self.case_id, case_metadata=self.case_metadata,
                                                        case_file_paths=self.case_file_paths, year=year)

    @property
    def survival_time(self, year=5):
        '''
        Return the survival time data.
        '''
        return self._get_survival_time_data(case_id=self.case_id, case_metadata=self.case_metadata, year=year)

    @property
    def primary_site(self):
        '''
        Return the primary site data.
        '''
        return self._get_primary_site_data(case_id=self.case_id, case_metadata=self.case_metadata)
