import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


TimmModel = collections.namedtuple('TimmModel', ['arch'])

model_ids_timm = {
    # CoAtNet
    'coatnet_0_rw_224.sw_in1k': TimmModel('coatnet'),
    'coatnet_1_rw_224.sw_in1k': TimmModel('coatnet'),
    # coatnet_2_rw_224.sw_in12k_ft_in1k
    'coatnet_bn_0_rw_224.sw_in1k': TimmModel('coatnet'),
    'coatnet_nano_rw_224.sw_in1k': TimmModel('coatnet'),
    # coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k
    'coatnet_rmlp_1_rw_224.sw_in1k': TimmModel('coatnet'),
    'coatnet_rmlp_2_rw_224.sw_in1k': TimmModel('coatnet'),
    # coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k
    # coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
    'coatnet_rmlp_nano_rw_224.sw_in1k': TimmModel('coatnet'),
    'coatnext_nano_rw_224.sw_in1k': TimmModel('coatnet'),

    # MaxViT
    'maxvit_base_tf_224.in1k': TimmModel('maxvit'),
    'maxvit_base_tf_384.in1k': TimmModel('maxvit'),
    'maxvit_base_tf_384.in21k_ft_in1k': TimmModel('maxvit'),
    'maxvit_base_tf_512.in1k': TimmModel('maxvit'),
    'maxvit_base_tf_512.in21k_ft_in1k': TimmModel('maxvit'),
    'maxvit_large_tf_224.in1k': TimmModel('maxvit'),
    'maxvit_large_tf_384.in1k': TimmModel('maxvit'),
    'maxvit_large_tf_384.in21k_ft_in1k': TimmModel('maxvit'),
    'maxvit_large_tf_512.in1k': TimmModel('maxvit'),
    'maxvit_large_tf_512.in21k_ft_in1k': TimmModel('maxvit'),
    'maxvit_nano_rw_256.sw_in1k': TimmModel('maxvit'),
    # maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k
    # maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k
    'maxvit_rmlp_nano_rw_256.sw_in1k': TimmModel('maxvit'),
    'maxvit_rmlp_pico_rw_256.sw_in1k': TimmModel('maxvit'),
    'maxvit_rmlp_small_rw_224.sw_in1k': TimmModel('maxvit'),
    'maxvit_rmlp_tiny_rw_256.sw_in1k': TimmModel('maxvit'),
    'maxvit_small_tf_224.in1k': TimmModel('maxvit'),
    'maxvit_small_tf_384.in1k': TimmModel('maxvit'),
    'maxvit_small_tf_512.in1k': TimmModel('maxvit'),
    'maxvit_tiny_rw_224.sw_in1k': TimmModel('maxvit'),
    'maxvit_tiny_tf_224.in1k': TimmModel('maxvit'),
    'maxvit_tiny_tf_384.in1k': TimmModel('maxvit'),
    'maxvit_tiny_tf_512.in1k': TimmModel('maxvit'),
    'maxvit_xlarge_tf_384.in21k_ft_in1k': TimmModel('maxvit'),
    'maxvit_xlarge_tf_512.in21k_ft_in1k': TimmModel('maxvit'),
    'maxxvit_rmlp_nano_rw_256.sw_in1k': TimmModel('maxvit'),
    'maxxvit_rmlp_small_rw_256.sw_in1k': TimmModel('maxvit'),
    'maxxvitv2_nano_rw_256.sw_in1k': TimmModel('maxvit'),
    # maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k
    # maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k

    # MobileViT
    'mobilevit_s.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevit_xs.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevit_xxs.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_050.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_075.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_100.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_125.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_150.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_150.cvnets_in22k_ft_in1k': TimmModel('mobilevit'),
    'mobilevitv2_150.cvnets_in22k_ft_in1k_384': TimmModel('mobilevit'),
    'mobilevitv2_175.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_175.cvnets_in22k_ft_in1k': TimmModel('mobilevit'),
    'mobilevitv2_175.cvnets_in22k_ft_in1k_384': TimmModel('mobilevit'),
    'mobilevitv2_200.cvnets_in1k': TimmModel('mobilevit'),
    'mobilevitv2_200.cvnets_in22k_ft_in1k': TimmModel('mobilevit'),
    'mobilevitv2_200.cvnets_in22k_ft_in1k_384': TimmModel('mobilevit'),

    # EdgeNeXt
    'edgenext_base.in21k_ft_in1k': TimmModel('edgenext'),
    'edgenext_base.usi_in1k': TimmModel('edgenext'),  # Distillation
    'edgenext_small.usi_in1k': TimmModel('edgenext'),  # Distillation
    'edgenext_small_rw.sw_in1k': TimmModel('edgenext'),
    'edgenext_x_small.in1k': TimmModel('edgenext'),
    'edgenext_xx_small.in1k': TimmModel('edgenext'),

    # PVT-v2 (Pyramid Vision Transformer)
    'pvt_v2_b0.in1k': TimmModel('pvt'),
    'pvt_v2_b1.in1k': TimmModel('pvt'),
    'pvt_v2_b2.in1k': TimmModel('pvt'),
    'pvt_v2_b2_li.in1k': TimmModel('pvt'),
    'pvt_v2_b3.in1k': TimmModel('pvt'),
    'pvt_v2_b4.in1k': TimmModel('pvt'),
    'pvt_v2_b5.in1k': TimmModel('pvt'),
}


def add_models(registry):
    for model_id in model_ids_timm:
        timm_model = model_ids_timm[model_id]
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if model_id.endswith('in21k_ft_in1k') or 'in22k_ft_in1k' in model_id:
            training_data = 'ImageNet-21k (14M)'
        else:
            assert 'in22k' not in model_id and 'in21k' not in model_id and 'in12k' not in model_id
            training_data = 'ImageNet-1k (1.2M)'
        if model_id.endswith('usi_in1k'):
            extra_annotations = ['Distillation']
        else:
            assert 'usi' not in model_id
            extra_annotations = []
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='hybrid',
                arch=timm_model.arch,
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=32 if model_id.startswith('coat') else 8,
                extra_annotations=extra_annotations
            )
        ))
