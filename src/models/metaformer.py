import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


model_ids = [
    # CAFormer
    'caformer_b36.sail_in1k',
    'caformer_b36.sail_in1k_384',
    'caformer_b36.sail_in22k_ft_in1k',
    'caformer_b36.sail_in22k_ft_in1k_384',
    'caformer_m36.sail_in1k',
    'caformer_m36.sail_in1k_384',
    'caformer_m36.sail_in22k_ft_in1k',
    'caformer_m36.sail_in22k_ft_in1k_384',
    'caformer_s18.sail_in1k',
    'caformer_s18.sail_in1k_384',
    'caformer_s18.sail_in22k_ft_in1k',
    'caformer_s18.sail_in22k_ft_in1k_384',
    'caformer_s36.sail_in1k',
    'caformer_s36.sail_in1k_384',
    'caformer_s36.sail_in22k_ft_in1k',
    'caformer_s36.sail_in22k_ft_in1k_384',

    # ConvFormer
    'convformer_b36.sail_in1k',
    'convformer_b36.sail_in1k_384',
    'convformer_b36.sail_in22k_ft_in1k',
    'convformer_b36.sail_in22k_ft_in1k_384',
    'convformer_m36.sail_in1k',
    'convformer_m36.sail_in1k_384',
    'convformer_m36.sail_in22k_ft_in1k',
    'convformer_m36.sail_in22k_ft_in1k_384',
    'convformer_s18.sail_in1k',
    'convformer_s18.sail_in1k_384',
    'convformer_s18.sail_in22k_ft_in1k',
    'convformer_s18.sail_in22k_ft_in1k_384',
    'convformer_s36.sail_in1k',
    'convformer_s36.sail_in1k_384',
    'convformer_s36.sail_in22k_ft_in1k',
    'convformer_s36.sail_in22k_ft_in1k_384',

    # PoolFormer
    'poolformer_m36.sail_in1k',
    'poolformer_m48.sail_in1k',
    'poolformer_s12.sail_in1k',
    'poolformer_s24.sail_in1k',
    'poolformer_s36.sail_in1k',
    'poolformerv2_m36.sail_in1k',
    'poolformerv2_m48.sail_in1k',
    'poolformerv2_s12.sail_in1k',
    'poolformerv2_s24.sail_in1k',
    'poolformerv2_s36.sail_in1k',
]


def add_models(registry):
    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        if model_id.startswith('caformer'):
            arch = 'caformer'
            arch_family = 'hybrid'
        elif model_id.startswith('convformer'):
            arch = 'convformer'
            arch_family = 'cnn'
        elif model_id.startswith('poolformer'):
            arch = 'poolformer'
            arch_family = 'mlp'
        else:
            assert False
        if 'in22k' in model_id:
            assert '.sail_in22k_ft_in1k' in model_id
            training_data = 'ImageNet-21k (14M)'
        else:
            assert 'in22k' not in model_id
            training_data = 'ImageNet-1k (1.2M)'
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family=arch_family,
                arch=arch,
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=8
            )
        ))
