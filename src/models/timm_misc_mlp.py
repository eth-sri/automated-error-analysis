import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


TimmModel = collections.namedtuple('TimmModel', ['arch'])

model_ids_timm = {
    'gmlp_s16_224.ra3_in1k': TimmModel('gmlp'),
    'gmixer_24_224.ra3_in1k': TimmModel('gmixer'),

    'resmlp_12_224.fb_distilled_in1k': TimmModel('resmlp'),
    'resmlp_12_224.fb_in1k': TimmModel('resmlp'),
    'resmlp_24_224.fb_distilled_in1k': TimmModel('resmlp'),
    'resmlp_24_224.fb_in1k': TimmModel('resmlp'),
    'resmlp_36_224.fb_distilled_in1k': TimmModel('resmlp'),
    'resmlp_36_224.fb_in1k': TimmModel('resmlp'),
    'resmlp_big_24_224.fb_distilled_in1k': TimmModel('resmlp'),
    'resmlp_big_24_224.fb_in1k': TimmModel('resmlp'),
    'resmlp_big_24_224.fb_in22k_ft_in1k': TimmModel('resmlp'),

    'mixer_b16_224.goog_in21k_ft_in1k': TimmModel('mlp-mixer'),
    'mixer_b16_224.miil_in21k_ft_in1k': TimmModel('mlp-mixer'),
    'mixer_l16_224.goog_in21k_ft_in1k': TimmModel('mlp-mixer'),
}


def add_models(registry):
    for model_id in model_ids_timm:
        timm_model = model_ids_timm[model_id]
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if model_id.endswith('_in21k_ft_in1k') or model_id.endswith('_in22k_ft_in1k'):
            training_data = 'ImageNet-21k (14M)'
        else:
            assert 'in22k' not in model_id and 'in21k' not in model_id
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
                arch_family='mlp',
                arch=timm_model.arch,
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=16,
                extra_annotations=extra_annotations
            )
        ))
