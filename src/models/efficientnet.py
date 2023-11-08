import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from torchvision.models import mnasnet0_5, MNASNet0_5_Weights
from torchvision.models import mnasnet0_75, MNASNet0_75_Weights
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights
from torchvision.models import mnasnet1_3, MNASNet1_3_Weights

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from src.models.model_base import ModelBase, ModelMetadata


TorchvisionModel = collections.namedtuple('TorchvisionModel', ['arch', 'model_cls', 'weights'])
TimmModel = collections.namedtuple('TimmModel', ['arch'])

model_ids_tv = {
    # MNASNet
    'torchvision/mnasnet0_5': TorchvisionModel('mnasnet', mnasnet0_5, MNASNet0_5_Weights),
    'torchvision/mnasnet0_75': TorchvisionModel('mnasnet', mnasnet0_75, MNASNet0_75_Weights),
    'torchvision/mnasnet1_0': TorchvisionModel('mnasnet', mnasnet1_0, MNASNet1_0_Weights),
    'torchvision/mnasnet1_3': TorchvisionModel('mnasnet', mnasnet1_3, MNASNet1_3_Weights),

    # MobileNet
    'torchvision/mobilenet_v2': TorchvisionModel('mobilenet', mobilenet_v2, MobileNet_V2_Weights),
    'torchvision/mobilenet_v3_small': TorchvisionModel('mobilenet', mobilenet_v3_small, MobileNet_V3_Small_Weights),
    'torchvision/mobilenet_v3_large': TorchvisionModel('mobilenet', mobilenet_v3_large, MobileNet_V3_Large_Weights),
}

model_ids_timm = {
    # MNASNet
    'mnasnet_100.rmsp_in1k': TimmModel('mnasnet'),
    'mnasnet_small.lamb_in1k': TimmModel('mnasnet'),
    'semnasnet_075.rmsp_in1k': TimmModel('mnasnet'),
    'semnasnet_100.rmsp_in1k': TimmModel('mnasnet'),

    # MobileNet
    'mobilenetv2_050.lamb_in1k': TimmModel('mobilenet'),
    'mobilenetv2_100.ra_in1k': TimmModel('mobilenet'),
    'mobilenetv2_110d.ra_in1k': TimmModel('mobilenet'),
    'mobilenetv2_120d.ra_in1k': TimmModel('mobilenet'),
    'mobilenetv2_140.ra_in1k': TimmModel('mobilenet'),
    'mobilenetv3_large_100.miil_in21k_ft_in1k': TimmModel('mobilenet'),  # ImageNet-21K
    'mobilenetv3_large_100.ra_in1k': TimmModel('mobilenet'),
    'mobilenetv3_rw.rmsp_in1k': TimmModel('mobilenet'),
    'mobilenetv3_small_050.lamb_in1k': TimmModel('mobilenet'),
    'mobilenetv3_small_075.lamb_in1k': TimmModel('mobilenet'),
    'mobilenetv3_small_100.lamb_in1k': TimmModel('mobilenet'),
    'tf_mobilenetv3_large_075.in1k': TimmModel('mobilenet'),
    'tf_mobilenetv3_large_100.in1k': TimmModel('mobilenet'),
    'tf_mobilenetv3_large_minimal_100.in1k': TimmModel('mobilenet'),
    'tf_mobilenetv3_small_075.in1k': TimmModel('mobilenet'),
    'tf_mobilenetv3_small_100.in1k': TimmModel('mobilenet'),
    'tf_mobilenetv3_small_minimal_100.in1k': TimmModel('mobilenet'),

    # FBNet
    'fbnetc_100.rmsp_in1k': TimmModel('fbnet'),
    'fbnetv3_b.ra2_in1k': TimmModel('fbnet'),
    'fbnetv3_d.ra2_in1k': TimmModel('fbnet'),
    'fbnetv3_g.ra2_in1k': TimmModel('fbnet'),

    # MixNet
    'mixnet_l.ft_in1k': TimmModel('mixnet'),
    'mixnet_m.ft_in1k': TimmModel('mixnet'),
    'mixnet_s.ft_in1k': TimmModel('mixnet'),
    'mixnet_xl.ra_in1k': TimmModel('mixnet'),

    'tf_mixnet_l.in1k': TimmModel('mixnet'),
    'tf_mixnet_m.in1k': TimmModel('mixnet'),
    'tf_mixnet_s.in1k': TimmModel('mixnet'),

    # TinyNet
    'tinynet_a.in1k': TimmModel('tinynet'),
    'tinynet_b.in1k': TimmModel('tinynet'),
    'tinynet_c.in1k': TimmModel('tinynet'),
    'tinynet_d.in1k': TimmModel('tinynet'),
    'tinynet_e.in1k': TimmModel('tinynet'),

    # LCNet
    'lcnet_050.ra2_in1k': TimmModel('lcnet'),
    'lcnet_075.ra2_in1k': TimmModel('lcnet'),
    'lcnet_100.ra2_in1k': TimmModel('lcnet'),

    # EfficientNet
    'efficientnet_b0.ra_in1k': TimmModel('efficientnet'),
    'efficientnet_b1.ft_in1k': TimmModel('efficientnet'),
    'efficientnet_b1_pruned.in1k': TimmModel('efficientnet'),
    'efficientnet_b2.ra_in1k': TimmModel('efficientnet'),
    'efficientnet_b2_pruned.in1k': TimmModel('efficientnet'),
    'efficientnet_b3.ra2_in1k': TimmModel('efficientnet'),
    'efficientnet_b3_pruned.in1k': TimmModel('efficientnet'),
    'efficientnet_b4.ra2_in1k': TimmModel('efficientnet'),
    # efficientnet_b5.sw_in12k
    # efficientnet_b5.sw_in12k_ft_in1k
    'efficientnet_el.ra_in1k': TimmModel('efficientnet'),
    'efficientnet_el_pruned.in1k': TimmModel('efficientnet'),
    'efficientnet_em.ra2_in1k': TimmModel('efficientnet'),
    'efficientnet_es.ra_in1k': TimmModel('efficientnet'),
    'efficientnet_es_pruned.in1k': TimmModel('efficientnet'),
    'efficientnet_lite0.ra_in1k': TimmModel('efficientnet'),
    'efficientnetv2_rw_m.agc_in1k': TimmModel('efficientnet'),
    'efficientnetv2_rw_s.ra2_in1k': TimmModel('efficientnet'),
    'efficientnetv2_rw_t.ra2_in1k': TimmModel('efficientnet'),

    'tf_efficientnet_b0.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b0.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b0.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b0.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b1.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b1.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b1.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b1.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b2.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b2.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b2.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b2.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b3.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b3.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b3.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b3.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b4.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b4.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b4.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b4.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b5.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b5.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b5.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b5.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b5.ra_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b6.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b6.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b6.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b7.aa_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b7.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b7.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b7.ra_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b8.ap_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_b8.ra_in1k': TimmModel('efficientnet'),

    'tf_efficientnet_cc_b0_4e.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_cc_b0_8e.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_cc_b1_8e.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_el.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_em.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_es.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_l2.ns_jft_in1k': TimmModel('efficientnet'),
    'tf_efficientnet_l2.ns_jft_in1k_475': TimmModel('efficientnet'),
    'tf_efficientnet_lite0.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_lite1.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_lite2.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_lite3.in1k': TimmModel('efficientnet'),
    'tf_efficientnet_lite4.in1k': TimmModel('efficientnet'),

    'tf_efficientnetv2_b0.in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_b1.in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_b2.in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_b3.in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_b3.in21k_ft_in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_l.in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_l.in21k_ft_in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_m.in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_m.in21k_ft_in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_s.in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_s.in21k_ft_in1k': TimmModel('efficientnet'),
    'tf_efficientnetv2_xl.in21k_ft_in1k': TimmModel('efficientnet'),

    'gc_efficientnetv2_rw_t.agc_in1k': TimmModel('efficientnet'),

    # ReXNet
    'rexnet_100.nav_in1k': TimmModel('rexnet'),
    'rexnet_130.nav_in1k': TimmModel('rexnet'),
    'rexnet_150.nav_in1k': TimmModel('rexnet'),
    'rexnet_200.nav_in1k': TimmModel('rexnet'),
    'rexnet_300.nav_in1k': TimmModel('rexnet'),
    # rexnetr_200.sw_in12k
    # rexnetr_200.sw_in12k_ft_in1k
    # rexnetr_300.sw_in12k
    # rexnetr_300.sw_in12k_ft_in1k

    # HardCoReNAS
    'hardcorenas_a.miil_green_in1k': TimmModel('hardcorenas'),
    'hardcorenas_b.miil_green_in1k': TimmModel('hardcorenas'),
    'hardcorenas_c.miil_green_in1k': TimmModel('hardcorenas'),
    'hardcorenas_d.miil_green_in1k': TimmModel('hardcorenas'),
    'hardcorenas_e.miil_green_in1k': TimmModel('hardcorenas'),
    'hardcorenas_f.miil_green_in1k': TimmModel('hardcorenas'),
}


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
                arch=torchvision_model.arch,
                model_id=model_id,
                training_data='ImageNet-1k (1.2M)',
                source='torchvision.models',
                eval_batch_size=32
            )
        ))

    for model_id in model_ids_timm:
        timm_model = model_ids_timm[model_id]
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        extra_annotations = []
        training_data = 'ImageNet-1k (1.2M)'
        if model_id.endswith('_pruned.in1k'):
            extra_annotations = ['Distillation']
        elif model_id.endswith('.ap_in1k'):
            extra_annotations = ['Adversarial Training']
        elif 'ns_jft' in model_id:
            extra_annotations = ['Semi-Supervised Learning']
            training_data = 'JFT (300M)'
        elif 'in21k' in model_id:
            training_data = 'ImageNet-21k (14M)'
        else:
            assert 'pruned' not in model_id
            assert 'ap' not in model_id
            assert 'ns' not in model_id and 'jft' not in model_id
            assert 'in21k' not in model_id
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch=timm_model.arch,
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=8,
                extra_annotations=extra_annotations
            )
        ))
