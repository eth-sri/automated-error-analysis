import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


model_ids = [
    'beit_base_patch16_224.in22k_ft_in22k_in1k',
    'beit_base_patch16_384.in22k_ft_in22k_in1k',
    'beit_large_patch16_224.in22k_ft_in22k_in1k',
    'beit_large_patch16_384.in22k_ft_in22k_in1k',
    'beit_large_patch16_512.in22k_ft_in22k_in1k',
    'beitv2_base_patch16_224.in1k_ft_in1k',
    'beitv2_base_patch16_224.in1k_ft_in22k_in1k',
    'beitv2_large_patch16_224.in1k_ft_in1k',
    'beitv2_large_patch16_224.in1k_ft_in22k_in1k',
]


def add_models(registry):
    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if 'in22k' in model_id:
            training_data = 'ImageNet-21k (14M)'
        else:
            assert 'in22k' not in model_id
            training_data = 'ImageNet-1k (1.2M)'
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch='beit',
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=8,
                extra_annotations=['Self-Supervised Learning']
            )
        ))
