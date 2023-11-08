import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


TimmModel = collections.namedtuple('TimmModel', ['arch'])

model_ids_timm = {
    # DLA
    'dla34.in1k': TimmModel('dla'),
    'dla46_c.in1k': TimmModel('dla'),
    'dla46x_c.in1k': TimmModel('dla'),
    'dla60.in1k': TimmModel('dla'),
    'dla60_res2net.in1k': TimmModel('res2net'),
    'dla60_res2next.in1k': TimmModel('res2net'),
    'dla60x.in1k': TimmModel('dla'),
    'dla60x_c.in1k': TimmModel('dla'),
    'dla102.in1k': TimmModel('dla'),
    'dla102x2.in1k': TimmModel('dla'),
    'dla102x.in1k': TimmModel('dla'),
    'dla169.in1k': TimmModel('dla'),

    # DPN (Dual Path Networks)
    'dpn68.mx_in1k': TimmModel('dpn'),
    'dpn68b.mx_in1k': TimmModel('dpn'),
    'dpn68b.ra_in1k': TimmModel('dpn'),
    'dpn92.mx_in1k': TimmModel('dpn'),
    'dpn98.mx_in1k': TimmModel('dpn'),
    'dpn107.mx_in1k': TimmModel('dpn'),
    'dpn131.mx_in1k': TimmModel('dpn'),

    # Xception
    'xception41.tf_in1k': TimmModel('xception'),
    'xception41p.ra3_in1k': TimmModel('xception'),
    'xception65.ra3_in1k': TimmModel('xception'),
    'xception65.tf_in1k': TimmModel('xception'),
    'xception65p.ra3_in1k': TimmModel('xception'),
    'xception71.tf_in1k': TimmModel('xception'),

    # NFNet
    'dm_nfnet_f0.dm_in1k': TimmModel('nfnet'),
    'dm_nfnet_f1.dm_in1k': TimmModel('nfnet'),
    'dm_nfnet_f2.dm_in1k': TimmModel('nfnet'),
    'dm_nfnet_f3.dm_in1k': TimmModel('nfnet'),
    'dm_nfnet_f4.dm_in1k': TimmModel('nfnet'),
    'dm_nfnet_f5.dm_in1k': TimmModel('nfnet'),
    'dm_nfnet_f6.dm_in1k': TimmModel('nfnet'),
    'eca_nfnet_l0.ra2_in1k': TimmModel('nfnet'),
    'eca_nfnet_l1.ra2_in1k': TimmModel('nfnet'),
    'eca_nfnet_l2.ra3_in1k': TimmModel('nfnet'),
    'nfnet_l0.ra2_in1k': TimmModel('nfnet'),
    'nf_regnet_b1.ra2_in1k': TimmModel('nfnet'),
    'nf_resnet50.ra2_in1k': TimmModel('nfnet'),

    # FocalNet
    'focalnet_base_lrf.ms_in1k': TimmModel('focalnet'),
    'focalnet_base_srf.ms_in1k': TimmModel('focalnet'),
    'focalnet_small_lrf.ms_in1k': TimmModel('focalnet'),
    'focalnet_small_srf.ms_in1k': TimmModel('focalnet'),
    'focalnet_tiny_lrf.ms_in1k': TimmModel('focalnet'),
    'focalnet_tiny_srf.ms_in1k': TimmModel('focalnet'),

    # HRNet
    'hrnet_w18.ms_aug_in1k': TimmModel('hrnet'),
    'hrnet_w18.ms_in1k': TimmModel('hrnet'),
    'hrnet_w18_small.ms_in1k': TimmModel('hrnet'),
    'hrnet_w18_small_v2.ms_in1k': TimmModel('hrnet'),
    'hrnet_w18_ssld.paddle_in1k': TimmModel('hrnet'),  # Distillation
    'hrnet_w30.ms_in1k': TimmModel('hrnet'),
    'hrnet_w32.ms_in1k': TimmModel('hrnet'),
    'hrnet_w40.ms_in1k': TimmModel('hrnet'),
    'hrnet_w44.ms_in1k': TimmModel('hrnet'),
    'hrnet_w48.ms_in1k': TimmModel('hrnet'),
    'hrnet_w48_ssld.paddle_in1k': TimmModel('hrnet'),  # Distillation
    'hrnet_w64.ms_in1k': TimmModel('hrnet'),

    # DarkNet
    'cs3darknet_focus_l.c2ns_in1k': TimmModel('darknet'),
    'cs3darknet_focus_m.c2ns_in1k': TimmModel('darknet'),
    'cs3darknet_l.c2ns_in1k': TimmModel('darknet'),
    'cs3darknet_m.c2ns_in1k': TimmModel('darknet'),
    'cs3darknet_x.c2ns_in1k': TimmModel('darknet'),
    'cs3sedarknet_l.c2ns_in1k': TimmModel('darknet'),
    'cs3sedarknet_x.c2ns_in1k': TimmModel('darknet'),
    'cspdarknet53.ra_in1k': TimmModel('darknet'),
    'darknet53.c2ns_in1k': TimmModel('darknet'),
    'darknetaa53.c2ns_in1k': TimmModel('darknet'),

    # ConvMixer
    'convmixer_768_32.in1k': TimmModel('convmixer'),
    'convmixer_1024_20_ks9_p14.in1k': TimmModel('convmixer'),
    'convmixer_1536_20.in1k': TimmModel('convmixer'),
}


def add_models(registry):
    for model_id in model_ids_timm:
        timm_model = model_ids_timm[model_id]
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        extra_annotations = []
        if timm_model.arch == 'hrnet':
            if model_id.endswith('_ssld.paddle_in1k'):
                extra_annotations = ['Distillation']
            else:
                assert 'ssld' not in model_id and 'paddle' not in model_id
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch=timm_model.arch,
                model_id=f'timm/{model_id}',
                training_data='ImageNet-1k (1.2M)',
                source='timm',
                eval_batch_size=32,
                extra_annotations=extra_annotations
            )
        ))
