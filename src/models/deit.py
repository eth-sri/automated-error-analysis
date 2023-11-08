import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


# A version of ViT.
model_ids = [
    'deit3_base_patch16_224.fb_in1k',
    'deit3_base_patch16_224.fb_in22k_ft_in1k',
    'deit3_base_patch16_384.fb_in1k',
    'deit3_base_patch16_384.fb_in22k_ft_in1k',
    'deit3_huge_patch14_224.fb_in1k',
    'deit3_huge_patch14_224.fb_in22k_ft_in1k',
    'deit3_large_patch16_224.fb_in1k',
    'deit3_large_patch16_224.fb_in22k_ft_in1k',
    'deit3_large_patch16_384.fb_in1k',
    'deit3_large_patch16_384.fb_in22k_ft_in1k',
    'deit3_medium_patch16_224.fb_in1k',
    'deit3_medium_patch16_224.fb_in22k_ft_in1k',
    'deit3_small_patch16_224.fb_in1k',
    'deit3_small_patch16_224.fb_in22k_ft_in1k',
    'deit3_small_patch16_384.fb_in1k',
    'deit3_small_patch16_384.fb_in22k_ft_in1k',
    'deit_base_distilled_patch16_224.fb_in1k',
    'deit_base_distilled_patch16_384.fb_in1k',
    'deit_base_patch16_224.fb_in1k',
    'deit_base_patch16_384.fb_in1k',
    'deit_small_distilled_patch16_224.fb_in1k',
    'deit_small_patch16_224.fb_in1k',
    'deit_tiny_distilled_patch16_224.fb_in1k',
    'deit_tiny_patch16_224.fb_in1k',
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
        if 'distilled' in model_id:
            extra_annotations = ['Distillation']
        else:
            assert 'dist' not in model_id
            extra_annotations = []
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch='deit',
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=4,
                extra_annotations=extra_annotations
            )
        ))
