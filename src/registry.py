from importlib import import_module
from typing import Dict

from loguru import logger

from src.models.model_base import ModelBase


ALL_MODULES = [
    'beit',
    'bit',
    'convnext',
    'deit',
    'efficientnet',
    'eva',
    'metaformer',
    'regnet',
    'resnet',
    'ssl_swsl_wsl',
    'swin',
    'timm_misc_cnn',
    'timm_misc_hybrid',
    'timm_misc_mlp',
    'timm_misc_transformer_dist',
    'timm_misc_transformer',
    'torchvision_misc',
    'vit',
    'xcit',
]


class ModelRegistry:

    def __init__(self):
        logger.info('Create ModelRegistry')
        self.models: Dict[str, ModelBase] = {}

    def add_model(self, model: ModelBase):
        model_id = model.model_metadata.model_id
        assert model_id not in self.models, f"Duplicate model {model_id} found. Model ids must be unique."
        logger.info(f'Add model {model_id}')
        self.models[model_id] = model

    def add_models_from_module(self, module_name: str):
        assert module_name in ALL_MODULES
        module = import_module(f'src.models.{module_name}')
        module.add_models(self)
