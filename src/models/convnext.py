import collections
from functools import partial

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

import torch.nn.functional as F
from transformers import ConvNextForImageClassification, ConvNextImageProcessor

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

from src.models.model_base import ModelBase, ModelMetadata


TorchvisionModel = collections.namedtuple('TorchvisionModel', ['model_cls', 'weights'])

tv_model_ids = {
    'torchvision/convnext_tiny': TorchvisionModel(convnext_tiny, ConvNeXt_Tiny_Weights),
    'torchvision/convnext_small': TorchvisionModel(convnext_small, ConvNeXt_Small_Weights),
    'torchvision/convnext_base': TorchvisionModel(convnext_base, ConvNeXt_Base_Weights),
    'torchvision/convnext_large': TorchvisionModel(convnext_large, ConvNeXt_Large_Weights),
}

fb_model_ids = [
    # ImageNet-1k
    'facebook/convnext-tiny-224',
    'facebook/convnext-small-224',
    'facebook/convnext-base-224',
    'facebook/convnext-base-384',
    'facebook/convnext-large-224',
    'facebook/convnext-large-384',

    # ImageNet-22k
    'facebook/convnext-base-224-22k-1k',
    'facebook/convnext-base-384-22k-1k',
    'facebook/convnext-large-384-22k-1k',
    'facebook/convnext-xlarge-224-22k-1k',
    'facebook/convnext-xlarge-384-22k-1k'
]

timm_model_ids = [
    'convnext_atto.d2_in1k',
    'convnext_atto_ols.a2_in1k',
    'convnext_femto.d1_in1k',
    'convnext_femto_ols.d1_in1k',
    'convnext_large.fb_in22k_ft_in1k',
    'convnext_nano.d1h_in1k',
    'convnext_nano_ols.d1h_in1k',
    'convnext_pico.d1_in1k',
    'convnext_pico_ols.d1_in1k',
    'convnext_small.fb_in22k_ft_in1k',
    'convnext_small.fb_in22k_ft_in1k_384',
    'convnext_tiny.fb_in22k_ft_in1k',
    'convnext_tiny.fb_in22k_ft_in1k_384',
    'convnext_tiny_hnf.a2h_in1k',

    # ImageNet-12k (a 11821 class subset of full ImageNet-22k)
    # convnext_nano.in12k_ft_in1k
    # convnext_small.in12k_ft_in1k
    # convnext_small.in12k_ft_in1k_384
    # convnext_tiny.in12k_ft_in1k
    # convnext_tiny.in12k_ft_in1k_384

    'convnext_base.clip_laion2b_augreg_ft_in1k',
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k',
    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384',
    # 'convnext_base.clip_laiona_augreg_ft_in1k_384',
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k',
    'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384',
    'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320',
    'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384',
    'convnext_xxlarge.clip_laion2b_soup_ft_in1k',

    'convnextv2_atto.fcmae_ft_in1k',
    'convnextv2_base.fcmae_ft_in1k',
    'convnextv2_base.fcmae_ft_in22k_in1k',
    'convnextv2_base.fcmae_ft_in22k_in1k_384',
    'convnextv2_femto.fcmae_ft_in1k',
    'convnextv2_huge.fcmae_ft_in1k',
    'convnextv2_huge.fcmae_ft_in22k_in1k_384',
    'convnextv2_huge.fcmae_ft_in22k_in1k_512',
    'convnextv2_large.fcmae_ft_in1k',
    'convnextv2_large.fcmae_ft_in22k_in1k',
    'convnextv2_large.fcmae_ft_in22k_in1k_384',
    'convnextv2_nano.fcmae_ft_in1k',
    'convnextv2_nano.fcmae_ft_in22k_in1k',
    'convnextv2_nano.fcmae_ft_in22k_in1k_384',
    'convnextv2_pico.fcmae_ft_in1k',
    'convnextv2_tiny.fcmae_ft_in1k',
    'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
]


def load_model(model_id):
    image_processor = ConvNextImageProcessor.from_pretrained(model_id)
    model = ConvNextForImageClassification.from_pretrained(model_id)
    return image_processor, model


def transform(image_processor, image):
    data = image_processor(image, return_tensors="pt")
    return data['pixel_values'].squeeze(0)


def classify_batch(model, inputs):
    logits = model(pixel_values=inputs).logits
    return F.softmax(logits, dim=1)


def add_models(registry):
    for model_id in tv_model_ids:
        torchvision_model = tv_model_ids[model_id]
        weights = torchvision_model.weights.DEFAULT
        model = torchvision_model.model_cls(weights=weights)
        registry.add_model(ModelBase(
            model=model,
            transform_fn=weights.transforms(),
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch='convnext',
                model_id=model_id,
                training_data='ImageNet-1k (1.2M)',
                source='torchvision.models',
                eval_batch_size=32
            )
        ))

    for model_id in fb_model_ids:
        image_processor, model = load_model(model_id)
        if '22k' in model_id:
            assert model_id.endswith('-22k-1k')
            training_data = 'ImageNet-21k (14M)'
            eval_batch_size=32 if model_id != 'facebook/convnext-xlarge-384-22k-1k' else 16
        else:
            assert '22k' not in model_id and not model_id.endswith('-22k-1k')
            training_data = 'ImageNet-1k (1.2M)'
            eval_batch_size = 64
        registry.add_model(ModelBase(
            model=model,
            transform_fn=partial(transform, image_processor),
            classify_batch_fn=classify_batch,
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch='convnext',
                model_id=model_id,
                training_data=training_data,
                source='huggingface',
                eval_batch_size=eval_batch_size
            )
        ))

    for model_id in timm_model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform_fn = create_transform(**config, is_training=False)
        extra_annotations = []
        training_data='ImageNet-1k (1.2M)'
        if model_id.startswith('convnextv2_'):
            assert '.fcmae_ft_' in model_id
            extra_annotations = ['Self-Supervised Learning']
            if '_in22k_' in model_id:
                training_data = 'ImageNet-21k (14M)'
        else:
            assert model_id.startswith('convnext_')
            if '.fb_in22k_ft_in1k' in model_id:
                training_data = 'ImageNet-21k (14M)'
            elif '.clip_laion2b' in model_id:
                extra_annotations = ['CLIP Training']
                training_data = 'LAION-2B (2.3B)'
            else:
                assert 'in22k' not in model_id
                assert 'in12k' not in model_id
                assert 'clip' not in model_id and 'laion' not in model_id
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform_fn,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch='convnext',
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=4,
                extra_annotations=extra_annotations
            )
        ))
