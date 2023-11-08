import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


model_ids = [
    # CoaT
    'coat_lite_medium.in1k',
    'coat_lite_medium_384.in1k',
    'coat_lite_mini.in1k',
    'coat_lite_small.in1k',
    'coat_lite_tiny.in1k',
    'coat_mini.in1k',
    'coat_small.in1k',
    'coat_tiny.in1k',

    # CrossViT
    'crossvit_9_240.in1k',
    'crossvit_9_dagger_240.in1k',
    'crossvit_15_240.in1k',
    'crossvit_15_dagger_240.in1k',
    'crossvit_15_dagger_408.in1k',
    'crossvit_18_240.in1k',
    'crossvit_18_dagger_240.in1k',
    'crossvit_18_dagger_408.in1k',
    'crossvit_base_240.in1k',
    'crossvit_small_240.in1k',
    'crossvit_tiny_240.in1k',

    # Twins
    'twins_pcpvt_base.in1k',
    'twins_pcpvt_large.in1k',
    'twins_pcpvt_small.in1k',
    'twins_svt_base.in1k',
    'twins_svt_large.in1k',
    'twins_svt_small.in1k',

    # ConViT
    'convit_base.fb_in1k',
    'convit_small.fb_in1k',
    'convit_tiny.fb_in1k',

    # DaViT
    'davit_base.msft_in1k',
    'davit_small.msft_in1k',
    'davit_tiny.msft_in1k',

    # FlexiViT
    'flexivit_base.300ep_in1k',
    'flexivit_base.600ep_in1k',
    'flexivit_base.1200ep_in1k',
    'flexivit_large.300ep_in1k',
    'flexivit_large.600ep_in1k',
    'flexivit_large.1200ep_in1k',
    'flexivit_small.300ep_in1k',
    'flexivit_small.600ep_in1k',
    'flexivit_small.1200ep_in1k',

    # NesT
    'nest_base_jx.goog_in1k',
    'nest_small_jx.goog_in1k',
    'nest_tiny_jx.goog_in1k',

    # MViT-v2
    'mvitv2_base.fb_in1k',
    'mvitv2_large.fb_in1k',
    'mvitv2_small.fb_in1k',
    'mvitv2_tiny.fb_in1k',

    # VOLO
    'volo_d1_224.sail_in1k',
    'volo_d1_384.sail_in1k',
    'volo_d2_224.sail_in1k',
    'volo_d2_384.sail_in1k',
    'volo_d3_224.sail_in1k',
    'volo_d3_448.sail_in1k',
    'volo_d4_224.sail_in1k',
    'volo_d4_448.sail_in1k',
    'volo_d5_224.sail_in1k',
    'volo_d5_448.sail_in1k',
    'volo_d5_512.sail_in1k',

    # GC ViT
    'gcvit_base.in1k',
    'gcvit_small.in1k',
    'gcvit_tiny.in1k',
    'gcvit_xtiny.in1k',
    'gcvit_xxtiny.in1k',
]


def add_models(registry):
    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if model_id.startswith('coat'):
            arch = 'coat'
            eval_batch_size = 32
        elif model_id.startswith('crossvit'):
            arch = 'crossvit'
            eval_batch_size = 16
        elif model_id.startswith('twins'):
            arch = 'twins'
            eval_batch_size = 16
        elif model_id.startswith('convit'):
            arch = 'convit'
            eval_batch_size = 32
        elif model_id.startswith('davit'):
            arch = 'davit'
            eval_batch_size = 32
        elif model_id.startswith('flexivit'):
            arch = 'flexivit'
            eval_batch_size = 32
        elif model_id.startswith('nest'):
            arch = 'nest'
            eval_batch_size = 32
        elif model_id.startswith('mvit'):
            arch = 'mvit'
            eval_batch_size = 32
        elif model_id.startswith('volo'):
            arch = 'volo'
            eval_batch_size = 16
        elif model_id.startswith('gcvit'):
            arch = 'gcvit'
            eval_batch_size = 32
        else:
            assert False
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch=arch,
                model_id=f'timm/{model_id}',
                training_data='ImageNet-1k (1.2M)',
                source='timm',
                eval_batch_size=eval_batch_size
            )
        ))
