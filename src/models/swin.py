import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from torchvision.models import swin_b, Swin_B_Weights
from torchvision.models import swin_s, Swin_S_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

from src.models.model_base import ModelBase, ModelMetadata

TorchvisionModel = collections.namedtuple('TorchvisionModel', ['model_cls', 'weights'])

model_ids_tv = {
    'swin_b': TorchvisionModel(swin_b, Swin_B_Weights),
    'swin_s': TorchvisionModel(swin_s, Swin_S_Weights),
    'swin_t': TorchvisionModel(swin_t, Swin_T_Weights),
    'swin_v2_b': TorchvisionModel(swin_v2_b, Swin_V2_B_Weights),
    'swin_v2_s': TorchvisionModel(swin_v2_s, Swin_V2_S_Weights),
    'swin_v2_t': TorchvisionModel(swin_v2_t, Swin_V2_T_Weights),
}

model_ids_timm = [
    'swin_base_patch4_window7_224.ms_in1k',
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_base_patch4_window12_384.ms_in1k',
    'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
    'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_large_patch4_window12_384.ms_in22k_ft_in1k',
    'swin_s3_base_224.ms_in1k',
    'swin_s3_small_224.ms_in1k',
    'swin_s3_tiny_224.ms_in1k',
    'swin_small_patch4_window7_224.ms_in1k',
    'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_tiny_patch4_window7_224.ms_in1k',
    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
    'swinv2_base_window8_256.ms_in1k',
    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    'swinv2_base_window16_256.ms_in1k',
    # swinv2_cr_small_224.sw_in1k
    # swinv2_cr_small_ns_224.sw_in1k
    # swinv2_cr_tiny_ns_224.sw_in1k
    'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
    'swinv2_small_window8_256.ms_in1k',
    'swinv2_small_window16_256.ms_in1k',
    'swinv2_tiny_window8_256.ms_in1k',
    'swinv2_tiny_window16_256.ms_in1k',
]


def add_models(registry):
    for model_id in model_ids_tv:
        torchvision_model = model_ids_tv[model_id]
        weights = torchvision_model.weights.DEFAULT
        model = torchvision_model.model_cls(weights=weights)
        model = model.eval()
        registry.add_model(ModelBase(
            model=model,
            transform_fn=weights.transforms(),
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch='swin',
                model_id=f'torchvision/{model_id}',
                training_data='ImageNet-1k (1.2M)',
                source='torchvision.models',
                eval_batch_size=16
            )
        ))

    for model_id in model_ids_timm:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if model_id.endswith('.ms_in22k_ft_in1k'):
            training_data = 'ImageNet-21k (14M)'
        else:
            assert model_id.endswith('.ms_in1k')
            assert 'in22k' not in model_id
            training_data = 'ImageNet-1k (1.2M)'
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch='swin',
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=4,
            )
        ))
