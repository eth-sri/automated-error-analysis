import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from torchvision.models import regnet_x_16gf, RegNet_X_16GF_Weights
from torchvision.models import regnet_x_1_6gf, RegNet_X_1_6GF_Weights
from torchvision.models import regnet_x_32gf, RegNet_X_32GF_Weights
from torchvision.models import regnet_x_3_2gf, RegNet_X_3_2GF_Weights
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights
from torchvision.models import regnet_x_8gf, RegNet_X_8GF_Weights
from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights
from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
from torchvision.models import regnet_y_800mf, RegNet_Y_800MF_Weights
from torchvision.models import regnet_y_8gf, RegNet_Y_8GF_Weights

from src.models.model_base import ModelBase, ModelMetadata

TorchvisionModel = collections.namedtuple('TorchvisionModel', ['model_cls', 'weights'])

model_ids_tv = {
    'regnet_x_16gf': TorchvisionModel(regnet_x_16gf, RegNet_X_16GF_Weights),
    'regnet_x_1_6gf': TorchvisionModel(regnet_x_1_6gf, RegNet_X_1_6GF_Weights),
    'regnet_x_32gf': TorchvisionModel(regnet_x_32gf, RegNet_X_32GF_Weights),
    'regnet_x_3_2gf': TorchvisionModel(regnet_x_3_2gf, RegNet_X_3_2GF_Weights),
    'regnet_x_400mf': TorchvisionModel(regnet_x_400mf, RegNet_X_400MF_Weights),
    'regnet_x_800mf': TorchvisionModel(regnet_x_800mf, RegNet_X_800MF_Weights),
    'regnet_x_8gf': TorchvisionModel(regnet_x_8gf, RegNet_X_8GF_Weights),
    'regnet_y_16gf': TorchvisionModel(regnet_y_16gf, RegNet_Y_16GF_Weights),
    'regnet_y_1_6gf': TorchvisionModel(regnet_y_1_6gf, RegNet_Y_1_6GF_Weights),
    'regnet_y_32gf': TorchvisionModel(regnet_y_32gf, RegNet_Y_32GF_Weights),
    'regnet_y_3_2gf': TorchvisionModel(regnet_y_3_2gf, RegNet_Y_3_2GF_Weights),
    'regnet_y_400mf': TorchvisionModel(regnet_y_400mf, RegNet_Y_400MF_Weights),
    'regnet_y_800mf': TorchvisionModel(regnet_y_800mf, RegNet_Y_800MF_Weights),
    'regnet_y_8gf': TorchvisionModel(regnet_y_8gf, RegNet_Y_8GF_Weights),
}

model_ids_tv_swag = {
    'regnet_y_128gf_swag': TorchvisionModel(regnet_y_128gf, RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1),
    'regnet_y_16gf_swag': TorchvisionModel(regnet_y_16gf, RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1),
    'regnet_y_32gf_swag': TorchvisionModel(regnet_y_32gf, RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
}

model_ids_timm = [
    'regnetv_040.ra3_in1k',
    'regnetv_064.ra3_in1k',
    'regnetx_002.pycls_in1k',
    'regnetx_004.pycls_in1k',
    'regnetx_006.pycls_in1k',
    'regnetx_008.pycls_in1k',
    'regnetx_016.pycls_in1k',
    'regnetx_032.pycls_in1k',
    'regnetx_040.pycls_in1k',
    'regnetx_064.pycls_in1k',
    'regnetx_080.pycls_in1k',
    'regnetx_120.pycls_in1k',
    'regnetx_160.pycls_in1k',
    'regnetx_320.pycls_in1k',
    'regnety_002.pycls_in1k',
    'regnety_004.pycls_in1k',
    'regnety_006.pycls_in1k',
    'regnety_008.pycls_in1k',
    'regnety_016.pycls_in1k',
    'regnety_032.pycls_in1k',
    'regnety_032.ra_in1k',
    'regnety_040.pycls_in1k',
    'regnety_040.ra3_in1k',
    'regnety_064.pycls_in1k',
    'regnety_064.ra3_in1k',
    'regnety_080.pycls_in1k',
    'regnety_080.ra3_in1k',
    'regnety_120.pycls_in1k',
    # regnety_120.sw_in12k_ft_in1k
    'regnety_160.deit_in1k',
    # regnety_160.lion_in12k_ft_in1k
    'regnety_160.pycls_in1k',
    # regnety_160.sw_in12k_ft_in1k
    'regnety_320.pycls_in1k',
    'regnety_320.seer_ft_in1k',
    'regnety_640.seer_ft_in1k',
    'regnety_1280.seer_ft_in1k',
    'regnety_2560.seer_ft_in1k',
    'regnetz_040.ra3_in1k',
    'regnetz_040_h.ra3_in1k',
    'regnetz_b16.ra3_in1k',
    'regnetz_c16.ra3_in1k',
    'regnetz_c16_evos.ch_in1k',
    'regnetz_d8.ra3_in1k',
    'regnetz_d8_evos.ch_in1k',
    'regnetz_d32.ra3_in1k',
    'regnetz_e8.ra3_in1k',
]


def add_models(registry):
    for model_id in model_ids_tv:
        torchvision_model = model_ids_tv[model_id]
        weights = torchvision_model.weights.DEFAULT
        model = torchvision_model.model_cls(weights=weights)
        registry.add_model(ModelBase(
            model=model,
            transform_fn=weights.transforms(),
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch='regnet',
                model_id=f'torchvision/{model_id}',
                training_data='ImageNet-1k (1.2M)',
                source='torchvision.models',
                eval_batch_size=32
            )
        ))

    for model_id in model_ids_tv_swag:
        torchvision_model = model_ids_tv_swag[model_id]
        weights = torchvision_model.weights
        model = torchvision_model.model_cls(weights=weights)
        registry.add_model(ModelBase(
            model=model,
            transform_fn=weights.transforms(),
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch='regnet',
                model_id=f'torchvision/{model_id}',
                training_data='Instagram (3.6B)',
                source='torchvision.models',
                eval_batch_size=16,
                extra_annotations=['Semi-Supervised Learning']
            )
        ))

    for model_id in model_ids_timm:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if model_id.endswith('.seer_ft_in1k'):
            training_data = 'RandomInternetImages (2B)'
            extra_annotations = ['Self-Supervised Learning']
        else:
            assert 'seer' not in model_id
            assert 'ft' not in model_id
            training_data = 'ImageNet-1k (1.2M)'
            extra_annotations = []
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch='regnet',
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=4,
                extra_annotations=extra_annotations
            )
        ))
