import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights

from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights

from src.models.model_base import ModelBase, ModelMetadata


TorchvisionModel = collections.namedtuple('TorchvisionModel', ['arch', 'model_cls', 'weights'])

model_ids_tv = {
    # ResNet
    'torchvision/resnet18': TorchvisionModel('resnet', resnet18, ResNet18_Weights),
    'torchvision/resnet34': TorchvisionModel('resnet', resnet34, ResNet34_Weights),
    'torchvision/resnet50': TorchvisionModel('resnet', resnet50, ResNet50_Weights),
    'torchvision/resnet101': TorchvisionModel('resnet', resnet101, ResNet101_Weights),
    'torchvision/resnet152': TorchvisionModel('resnet', resnet152, ResNet152_Weights),

    # ResNeXt
    'torchvision/resnext50_32x4d': TorchvisionModel('resnext', resnext50_32x4d, ResNeXt50_32X4D_Weights),
    'torchvision/resnext101_32x8d': TorchvisionModel('resnext', resnext101_32x8d, ResNeXt101_32X8D_Weights),
    'torchvision/resnext101_64x4d': TorchvisionModel('resnext', resnext101_64x4d, ResNeXt101_64X4D_Weights),

    # Wide ResNet
    'torchvision/wide_resnet50_2': TorchvisionModel('wide_resnet', wide_resnet50_2, Wide_ResNet50_2_Weights),
    'torchvision/wide_resnet101_2': TorchvisionModel('wide_resnet', wide_resnet101_2, Wide_ResNet101_2_Weights),
}

model_ids_timm = [
    # ResNet
    'resnet10t.c3_in1k',
    'resnet14t.c3_in1k',
    'resnet18.a1_in1k',
    'resnet18.a2_in1k',
    'resnet18.a3_in1k',
    'resnet18.gluon_in1k',
    'resnet18d.ra2_in1k',
    'resnet26.bt_in1k',
    'resnet26d.bt_in1k',
    'resnet26t.ra2_in1k',
    'resnet32ts.ra2_in1k',
    'resnet33ts.ra2_in1k',
    'resnet34.a1_in1k',
    'resnet34.a2_in1k',
    'resnet34.a3_in1k',
    'resnet34.bt_in1k',
    'resnet34.gluon_in1k',
    'resnet34d.ra2_in1k',
    'resnet50.a1_in1k',
    'resnet50.a1h_in1k',
    'resnet50.a2_in1k',
    'resnet50.a3_in1k',
    'resnet50.am_in1k',
    'resnet50.b1k_in1k',
    'resnet50.b2k_in1k',
    'resnet50.bt_in1k',
    'resnet50.c1_in1k',
    'resnet50.c2_in1k',
    'resnet50.d_in1k',
    'resnet50.gluon_in1k',
    'resnet50.ra_in1k',
    'resnet50.ram_in1k',
    'resnet50_gn.a1h_in1k',
    'resnet50c.gluon_in1k',
    'resnet50d.a1_in1k',
    'resnet50d.a2_in1k',
    'resnet50d.a3_in1k',
    'resnet50d.gluon_in1k',
    'resnet50d.ra2_in1k',
    'resnet50s.gluon_in1k',
    'resnet51q.ra2_in1k',
    'resnet61q.ra2_in1k',
    'resnet101.a1_in1k',
    'resnet101.a1h_in1k',
    'resnet101.a2_in1k',
    'resnet101.a3_in1k',
    'resnet101.gluon_in1k',
    'resnet101c.gluon_in1k',
    'resnet101d.gluon_in1k',
    'resnet101d.ra2_in1k',
    'resnet101s.gluon_in1k',
    'resnet152.a1_in1k',
    'resnet152.a1h_in1k',
    'resnet152.a2_in1k',
    'resnet152.a3_in1k',
    'resnet152.gluon_in1k',
    'resnet152c.gluon_in1k',
    'resnet152d.gluon_in1k',
    'resnet152d.ra2_in1k',
    'resnet152s.gluon_in1k',
    'resnet200d.ra2_in1k',
    'resnetaa50.a1h_in1k',

    # resnetaa50d.d_in12k
    # resnetaa50d.sw_in12k
    # resnetaa50d.sw_in12k_ft_in1k
    # resnetaa101d.sw_in12k
    # resnetaa101d.sw_in12k_ft_in1k

    'resnetblur50.bt_in1k',
    'resnetrs50.tf_in1k',
    'resnetrs101.tf_in1k',
    'resnetrs152.tf_in1k',
    'resnetrs200.tf_in1k',
    'resnetrs270.tf_in1k',
    'resnetrs350.tf_in1k',
    'resnetrs420.tf_in1k',
    'resnetv2_50.a1h_in1k',
    'resnetv2_50d_evos.ah_in1k',
    'resnetv2_50d_gn.ah_in1k',
    'resnetv2_101.a1h_in1k',

    'cspresnet50.ra_in1k',
    'gcresnet33ts.ra2_in1k',
    'gcresnet50t.ra2_in1k',

    # LambdaNet (based on ResNet)
    'lambda_resnet26rpt_256.c1_in1k',
    'lambda_resnet26t.c1_in1k',
    'lambda_resnet50ts.a1h_in1k',

    # ResNeXt
    'resnext26ts.ra2_in1k',
    'resnext50_32x4d.a1_in1k',
    'resnext50_32x4d.a1h_in1k',
    'resnext50_32x4d.a2_in1k',
    'resnext50_32x4d.a3_in1k',
    'resnext50_32x4d.gluon_in1k',
    'resnext50_32x4d.ra_in1k',
    'resnext50d_32x4d.bt_in1k',
    'resnext101_32x4d.gluon_in1k',
    'resnext101_64x4d.c1_in1k',
    'resnext101_64x4d.gluon_in1k',

    'bat_resnext26ts.ch_in1k',
    'cspresnext50.ra_in1k',
    'gcresnext26ts.ch_in1k',
    'gcresnext50ts.ch_in1k',

    # Efficient Channel Attentions ResNets / ResNeXts
    'ecaresnet26t.ra2_in1k',
    'ecaresnet50d.miil_in1k',
    'ecaresnet50d_pruned.miil_in1k',  # Distillation
    'ecaresnet50t.a1_in1k',
    'ecaresnet50t.a2_in1k',
    'ecaresnet50t.a3_in1k',
    'ecaresnet50t.ra2_in1k',
    'ecaresnet101d.miil_in1k',
    'ecaresnet101d_pruned.miil_in1k',  # Distillation
    'ecaresnet269d.ra2_in1k',
    'ecaresnetlight.miil_in1k',
    'eca_resnet33ts.ra2_in1k',
    'eca_resnext26ts.ch_in1k',

    # Squeeze-Excitation ResNets
    'seresnet33ts.ra2_in1k',
    'seresnet50.a1_in1k',
    'seresnet50.a2_in1k',
    'seresnet50.a3_in1k',
    'seresnet50.ra2_in1k',
    'seresnet152d.ra2_in1k',

    # Squeeze-Excitation ResNeXts
    'seresnext26d_32x4d.bt_in1k',
    'seresnext26t_32x4d.bt_in1k',
    'seresnext26ts.ch_in1k',
    'seresnext50_32x4d.gluon_in1k',
    'seresnext50_32x4d.racm_in1k',
    'seresnext101_32x4d.gluon_in1k',
    'seresnext101_32x8d.ah_in1k',
    'seresnext101_64x4d.gluon_in1k',
    'seresnext101d_32x8d.ah_in1k',
    'seresnextaa101d_32x8d.ah_in1k',
    # seresnextaa101d_32x8d.sw_in12k
    # seresnextaa101d_32x8d.sw_in12k_ft_in1k
    # seresnextaa101d_32x8d.sw_in12k_ft_in1k_288

    # Res2Net
    'res2net50_14w_8s.in1k',
    'res2net50_26w_4s.in1k',
    'res2net50_26w_6s.in1k',
    'res2net50_26w_8s.in1k',
    'res2net50_48w_2s.in1k',
    'res2net50d.in1k',
    'res2net101_26w_4s.in1k',
    'res2net101d.in1k',
    'res2next50.in1k',

    # ResNeSt
    'resnest14d.gluon_in1k',
    'resnest26d.gluon_in1k',
    'resnest50d.in1k',
    'resnest50d_1s4x24d.in1k',
    'resnest50d_4s2x40d.in1k',
    'resnest101e.in1k',
    'resnest200e.in1k',
    'resnest269e.in1k',

    # TResNet (High Performance GPU-Dedicated Architecture)
    'tresnet_l.miil_in1k',
    'tresnet_l.miil_in1k_448',
    'tresnet_m.miil_in1k',
    'tresnet_m.miil_in1k_448',
    'tresnet_m.miil_in21k_ft_in1k',
    'tresnet_v2_l.miil_in21k_ft_in1k',
    'tresnet_xl.miil_in1k',
    'tresnet_xl.miil_in1k_448',

    # Wide ResNet
    'wide_resnet50_2.racm_in1k',

    # HaloNet
    'eca_halonext26ts.c1_in1k',
    'halonet26t.a1h_in1k',
    'halonet50ts.a1h_in1k',
    'sehalonet33ts.ra2_in1k',

    # SKNet
    'skresnet18.ra_in1k',
    'skresnet34.ra_in1k',
    'skresnext50_32x4d.ra_in1k',
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
                arch=torchvision_model.arch,
                model_id=model_id,
                training_data='ImageNet-1k (1.2M)',
                source='torchvision.models',
                eval_batch_size=32
            )
        ))

    for model_id in model_ids_timm:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        training_data = 'ImageNet-1k (1.2M)'
        # ResNet
        if model_id.startswith('resnet') or model_id.startswith('cspresnet') or model_id.startswith('gcresnet'):
            arch = 'resnet'
        elif model_id.startswith('ecaresnet') or model_id.startswith('eca_resne'):
            arch = 'ecaresnet'
        elif model_id.startswith('seresnet'):
            arch = 'seresnet'
        elif model_id.startswith('lambda_resnet'):
            arch = 'lambda_resnet'
        elif model_id.startswith('wide_resnet'):
            arch = 'wide_resnet'
        elif model_id.startswith('skresne'):
            arch = 'skresnet'
        # ResNeXt
        elif model_id.startswith('resnext') or model_id.startswith('gcresnext') or model_id.startswith('bat_resnext') \
                or model_id.startswith('cspresnext'):
            arch = 'resnext'
        elif model_id.startswith('seresnext'):
            arch = 'seresnext'
        # The rest ...
        elif model_id.startswith('res2net') or model_id.startswith('res2next'):
            arch = 'res2net'
        elif model_id.startswith('resnest'):
            arch = 'resnest'
        elif model_id.startswith('tresnet'):
            arch = 'tresnet'
            if model_id.endswith('in21k_ft_in1k'):
                training_data = 'ImageNet-21k (14M)'
            else:
                assert 'in21k' not in model_id
        elif 'halone' in model_id:
            arch = 'halonet'
        else:
            assert False
        if 'pruned' in model_id:
            extra_annotations = ['Distillation']
        else:
            extra_annotations = []
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch=arch,
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=32,
                extra_annotations=extra_annotations
            )
        ))
