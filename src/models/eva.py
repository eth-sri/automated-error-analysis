import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


model_ids = [
    'eva_giant_patch14_224.clip_ft_in1k',
    'eva_giant_patch14_336.clip_ft_in1k',

    'eva_giant_patch14_336.m30m_ft_in22k_in1k',
    'eva_giant_patch14_560.m30m_ft_in22k_in1k',
    'eva_large_patch14_196.in22k_ft_in1k',
    'eva_large_patch14_196.in22k_ft_in22k_in1k',
    'eva_large_patch14_336.in22k_ft_in1k',
    'eva_large_patch14_336.in22k_ft_in22k_in1k',

    'eva02_base_patch14_448.mim_in22k_ft_in1k',
    'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
    'eva02_large_patch14_448.mim_in22k_ft_in1k',
    'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k',
    'eva02_large_patch14_448.mim_m38m_ft_in1k',
    'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
    'eva02_small_patch14_336.mim_in22k_ft_in1k',
    'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
]


def add_models(registry):
    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if 'clip' in model_id:
            training_data = 'LAION-400M (400M)'
        elif 'm38m' in model_id:
            training_data = 'Merged (38M)'
        elif 'm30m' in model_id:
            training_data = 'Merged (30M)'
        elif 'in22k' in model_id:
            assert 'm38m' not in model_id and 'm30m' not in model_id and 'clip' not in model_id
            training_data = 'ImageNet-21k (14M)'
        else:
            assert False
        extra_annotations = ['Self-Supervised Learning']
        if 'clip' in model_id:
            extra_annotations.append('CLIP Training')
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch='eva',
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=4 if 'giant' in model_id else 8,
                extra_annotations=extra_annotations
            )
        ))
