import re
from abc import abstractmethod

import numpy as np
import torch.nn as nn
import yaml


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __structure__(self):
        """
        Model structure in dict format.
        """
        line_str = []
        raw_str = super().__str__()

        # For GraphConv.
        raw_str = re.sub(r'(GraphConv\()\n\s+(.*)\n\s+(.*)\n\s+(\))', r'\1\2\3\4', raw_str)
        raw_str = re.sub(r'\(_activation\): ', ', activation=', raw_str)
        
        # Fix TransformerEncoderLayer representation
        raw_str = re.sub(r'(\d+) x TransformerEncoderLayer:', r'TransformerEncoderLayers: \1', raw_str)
        
        for line in raw_str.split('\n'):
            # "Sequential" and "(\d+)" form a list.
            line = re.sub(r'Sequential\(', r'', line)
            line = re.sub(r'  \(\d+\):', '-', line)
            # Remove brackets.
            line = re.sub(r'\((\w+)\):', r'\1:', line)
            # Line ending with "(" is the start of a new dict.
            line = re.sub(r'\($', ': ', line)
            # Remove redundant keys.
            line = re.sub(r'\w+?: (\w+):', r'\1:', line)
            # Remove closing brackets (YAML does not need them).
            line = re.sub(r'(^|\s)\)$', '', line)
            
            # Skip problematic lines entirely
            if 'TransformerEncoderLayer:' in line and ':' in line.split('TransformerEncoderLayer:')[1]:
                continue
                
            if line.strip() != '':
                line_str.append(line)
        
        # For debugging
        # print('\n'.join(line_str))
        
        try:
            model_struct: dict = yaml.safe_load('\n'.join(line_str))
            return model_struct
        except yaml.YAMLError as e:
            print(f"YAML Error: {e}")
            # As a fallback, return a simple dictionary
            return {"model_type": self.__class__.__name__, "error": "Could not parse model structure"}
