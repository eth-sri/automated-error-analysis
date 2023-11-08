import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


model_ids = [
    'xcit_large_24_p8_224.fb_dist_in1k',
    'xcit_large_24_p8_224.fb_in1k',
    'xcit_large_24_p8_384.fb_dist_in1k',
    'xcit_large_24_p16_224.fb_dist_in1k',
    'xcit_large_24_p16_224.fb_in1k',
    'xcit_large_24_p16_384.fb_dist_in1k',
    'xcit_medium_24_p8_224.fb_dist_in1k',
    'xcit_medium_24_p8_224.fb_in1k',
    'xcit_medium_24_p8_384.fb_dist_in1k',
    'xcit_medium_24_p16_224.fb_dist_in1k',
    'xcit_medium_24_p16_224.fb_in1k',
    'xcit_medium_24_p16_384.fb_dist_in1k',
    'xcit_nano_12_p8_224.fb_dist_in1k',
    'xcit_nano_12_p8_224.fb_in1k',
    'xcit_nano_12_p8_384.fb_dist_in1k',
    'xcit_nano_12_p16_224.fb_dist_in1k',
    'xcit_nano_12_p16_224.fb_in1k',
    'xcit_nano_12_p16_384.fb_dist_in1k',
    'xcit_small_12_p8_224.fb_dist_in1k',
    'xcit_small_12_p8_224.fb_in1k',
    'xcit_small_12_p8_384.fb_dist_in1k',
    'xcit_small_12_p16_224.fb_dist_in1k',
    'xcit_small_12_p16_224.fb_in1k',
    'xcit_small_12_p16_384.fb_dist_in1k',
    'xcit_small_24_p8_224.fb_dist_in1k',
    'xcit_small_24_p8_224.fb_in1k',
    'xcit_small_24_p8_384.fb_dist_in1k',
    'xcit_small_24_p16_224.fb_dist_in1k',
    'xcit_small_24_p16_224.fb_in1k',
    'xcit_small_24_p16_384.fb_dist_in1k',
    'xcit_tiny_12_p8_224.fb_dist_in1k',
    'xcit_tiny_12_p8_224.fb_in1k',
    'xcit_tiny_12_p8_384.fb_dist_in1k',
    'xcit_tiny_12_p16_224.fb_dist_in1k',
    'xcit_tiny_12_p16_224.fb_in1k',
    'xcit_tiny_12_p16_384.fb_dist_in1k',
    'xcit_tiny_24_p8_224.fb_dist_in1k',
    'xcit_tiny_24_p8_224.fb_in1k',
    'xcit_tiny_24_p8_384.fb_dist_in1k',
    'xcit_tiny_24_p16_224.fb_dist_in1k',
    'xcit_tiny_24_p16_224.fb_in1k',
    'xcit_tiny_24_p16_384.fb_dist_in1k',
]


def add_models(registry):
    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if model_id.endswith('.fb_dist_in1k'):
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
                arch='xcit',
                model_id=f'timm/{model_id}',
                training_data='ImageNet-1k (1.2M)',
                source='timm',
                eval_batch_size=8,
                extra_annotations=extra_annotations
            )
        ))
