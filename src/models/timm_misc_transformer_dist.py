import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


model_ids = [
    # CaiT (a version of ViT)
    'cait_m36_384.fb_dist_in1k',
    'cait_m48_448.fb_dist_in1k',
    'cait_s24_224.fb_dist_in1k',
    'cait_s24_384.fb_dist_in1k',
    'cait_s36_384.fb_dist_in1k',
    'cait_xs24_384.fb_dist_in1k',
    'cait_xxs24_224.fb_dist_in1k',
    'cait_xxs24_384.fb_dist_in1k',
    'cait_xxs36_224.fb_dist_in1k',
    'cait_xxs36_384.fb_dist_in1k',

    # LeViT
    'levit_128.fb_dist_in1k',
    'levit_128s.fb_dist_in1k',
    'levit_192.fb_dist_in1k',
    'levit_256.fb_dist_in1k',
    'levit_384.fb_dist_in1k',
    'levit_conv_128.fb_dist_in1k',
    'levit_conv_128s.fb_dist_in1k',
    'levit_conv_192.fb_dist_in1k',
    'levit_conv_256.fb_dist_in1k',
    'levit_conv_384.fb_dist_in1k',

    # PiT
    'pit_b_224.in1k',
    'pit_b_distilled_224.in1k',
    'pit_s_224.in1k',
    'pit_s_distilled_224.in1k',
    'pit_ti_224.in1k',
    'pit_ti_distilled_224.in1k',
    'pit_xs_224.in1k',
    'pit_xs_distilled_224.in1k',

    # EfficientFormer
    'efficientformer_l1.snap_dist_in1k',
    'efficientformer_l3.snap_dist_in1k',
    'efficientformer_l7.snap_dist_in1k',
    'efficientformerv2_l.snap_dist_in1k',
    'efficientformerv2_s0.snap_dist_in1k',
    'efficientformerv2_s1.snap_dist_in1k',
    'efficientformerv2_s2.snap_dist_in1k',
]


def add_models(registry):
    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        extra_annotations = ['Distillation']
        if model_id.startswith('cait'):
            arch = 'cait'
            arch_family = 'transformer'
        elif model_id.startswith('levit'):
            arch = 'levit'
            arch_family = 'hybrid'
        elif model_id.startswith('pit'):
            arch = 'pit'
            arch_family = 'transformer'
            if 'distilled' not in model_id:
                extra_annotations = []
        elif model_id.startswith('efficientformer'):
            arch = 'efficientformer'
            arch_family = 'hybrid'
        else:
            assert False
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family=arch_family,
                arch=arch,
                model_id=f'timm/{model_id}',
                training_data='ImageNet-1k (1.2M)',
                source='timm',
                eval_batch_size=16,
                extra_annotations=extra_annotations
            )
        ))
