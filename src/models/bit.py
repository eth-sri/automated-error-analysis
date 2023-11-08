import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from src.models.model_base import ModelBase, ModelMetadata


model_ids = [
    'resnetv2_50x1_bit.goog_in21k_ft_in1k',
    'resnetv2_50x3_bit.goog_in21k_ft_in1k',
    'resnetv2_101x1_bit.goog_in21k_ft_in1k',
    'resnetv2_101x3_bit.goog_in21k_ft_in1k',
    'resnetv2_152x2_bit.goog_in21k_ft_in1k',
    'resnetv2_152x4_bit.goog_in21k_ft_in1k',
    'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k',
    'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384',
    'resnetv2_50x1_bit.goog_distilled_in1k'  # Distilled from ImageNet-21k pretrained teacher model on ImageNet-1k
]


def add_models(registry):
    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        extra_annotations = []
        if model_id == 'resnetv2_50x1_bit.goog_distilled_in1k':
            extra_annotations = ['Distillation']
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch='resnet',
                model_id=f'timm/{model_id}',
                training_data='ImageNet-21k (14M)',
                source='timm',
                eval_batch_size=8,
                extra_annotations=extra_annotations
            )
        ))
