import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.models import vit_l_32, ViT_L_32_Weights

from src.models.model_base import ModelBase, ModelMetadata


TorchvisionModel = collections.namedtuple('TorchvisionModel', ['model_cls', 'weights', 'extra_annotations'])

model_ids_tv = {
    'vit_b_16': TorchvisionModel(vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, []),
    'vit_b_16_swag': TorchvisionModel(vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, ['Semi-Supervised Learning']),
    'vit_b_32': TorchvisionModel(vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, []),
    'vit_h_14_swag': TorchvisionModel(vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1, ['Semi-Supervised Learning']),
    'vit_l_16': TorchvisionModel(vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1, []),
    'vit_l_16_swag': TorchvisionModel(vit_l_16, ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1, ['Semi-Supervised Learning']),
    'vit_l_32': TorchvisionModel(vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1, []),
}

model_ids = [
    'vit_base_patch8_224.augreg2_in21k_ft_in1k',
    'vit_base_patch8_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',
    'vit_base_patch16_224.augreg_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
    'vit_base_patch16_224.orig_in21k_ft_in1k',
    'vit_base_patch16_224.sam_in1k',
    'vit_base_patch16_224_miil.in21k_ft_in1k',
    'vit_base_patch16_384.augreg_in1k',
    'vit_base_patch16_384.augreg_in21k_ft_in1k',
    'vit_base_patch16_384.orig_in21k_ft_in1k',
    'vit_base_patch32_224.augreg_in1k',
    'vit_base_patch32_224.augreg_in21k_ft_in1k',
    'vit_base_patch32_224.sam_in1k',
    'vit_base_patch32_384.augreg_in1k',
    'vit_base_patch32_384.augreg_in21k_ft_in1k',
    'vit_large_patch16_224.augreg_in21k_ft_in1k',
    'vit_large_patch16_384.augreg_in21k_ft_in1k',
    'vit_large_patch32_384.orig_in21k_ft_in1k',
    'vit_small_patch16_224.augreg_in1k',
    'vit_small_patch16_224.augreg_in21k_ft_in1k',
    'vit_small_patch16_384.augreg_in1k',
    'vit_small_patch16_384.augreg_in21k_ft_in1k',
    'vit_small_patch32_224.augreg_in21k_ft_in1k',
    'vit_small_patch32_384.augreg_in21k_ft_in1k',
    'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
    'vit_tiny_patch16_384.augreg_in21k_ft_in1k',

    'vit_base_patch16_rpn_224.sw_in1k',

    # ImageNet-12k (a 11821 class subset of full ImageNet-22k)
    # vit_medium_patch16_gap_256.sw_in12k_ft_in1k
    # vit_medium_patch16_gap_384.sw_in12k_ft_in1k

    # ResNet - Vision Transformer (ViT) hybrid models
    'vit_base_r50_s16_384.orig_in21k_ft_in1k',
    'vit_large_r50_s32_224.augreg_in21k_ft_in1k',
    'vit_large_r50_s32_384.augreg_in21k_ft_in1k',
    'vit_small_r26_s32_224.augreg_in21k_ft_in1k',
    'vit_small_r26_s32_384.augreg_in21k_ft_in1k',
    'vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k',
    'vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k',

    # Experimental / WIP:
    'vit_relpos_base_patch16_224.sw_in1k',
    'vit_relpos_base_patch16_clsgap_224.sw_in1k',
    'vit_relpos_base_patch32_plus_rpn_256.sw_in1k',
    'vit_relpos_medium_patch16_224.sw_in1k',
    'vit_relpos_medium_patch16_cls_224.sw_in1k',
    'vit_relpos_medium_patch16_rpn_224.sw_in1k',
    'vit_relpos_small_patch16_224.sw_in1k',
    'vit_srelpos_medium_patch16_224.sw_in1k',
    'vit_srelpos_small_patch16_224.sw_in1k',

    # CLIP training
    'vit_base_patch16_clip_224.laion2b_ft_in1k',
    'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',
    'vit_base_patch16_clip_224.openai_ft_in1k',
    'vit_base_patch16_clip_224.openai_ft_in12k_in1k',
    'vit_base_patch16_clip_384.laion2b_ft_in1k',
    'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
    'vit_base_patch16_clip_384.openai_ft_in1k',
    'vit_base_patch16_clip_384.openai_ft_in12k_in1k',
    'vit_base_patch32_clip_224.laion2b_ft_in1k',
    'vit_base_patch32_clip_224.laion2b_ft_in12k_in1k',
    'vit_base_patch32_clip_224.openai_ft_in1k',
    'vit_base_patch32_clip_384.laion2b_ft_in12k_in1k',
    'vit_base_patch32_clip_384.openai_ft_in12k_in1k',
    'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k',
    'vit_huge_patch14_clip_224.laion2b_ft_in1k',
    'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
    'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k',
    'vit_large_patch14_clip_224.laion2b_ft_in1k',
    'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k',
    'vit_large_patch14_clip_224.openai_ft_in1k',
    'vit_large_patch14_clip_224.openai_ft_in12k_in1k',
    'vit_large_patch14_clip_336.laion2b_ft_in1k',
    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k',
    'vit_large_patch14_clip_336.openai_ft_in12k_in1k',
]


def add_models(registry):
    for model_id in model_ids_tv:
        torchvision_model = model_ids_tv[model_id]
        weights = torchvision_model.weights
        model = torchvision_model.model_cls(weights=weights)
        model = model.eval()
        extra_annotations = torchvision_model.extra_annotations
        if not extra_annotations:
            assert len(extra_annotations) == 0
            training_data = 'ImageNet-1k (1.2M)'
        else:
            assert extra_annotations == ['Semi-Supervised Learning']
            training_data = 'Instagram (3.6B)'
        registry.add_model(ModelBase(
            model=model,
            transform_fn=weights.transforms(),
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch='vit',
                model_id=f'torchvision/{model_id}',
                training_data=training_data,
                source='torchvision.models',
                eval_batch_size=2,
                extra_annotations=extra_annotations
            )
        ))

    for model_id in model_ids:
        model = timm.create_model(model_id, pretrained=True)
        model = model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        extra_annotations = []
        training_data = 'ImageNet-1k (1.2M)'
        if 'clip' in model_id:
            extra_annotations = ['CLIP Training']
            if '.laion2b_ft_' in model_id:
                training_data = 'LAION-2B (2.3B)'
            else:
                assert 'laion2b' not in model_id
                assert '.openai_ft_' in model_id
                training_data = 'WIT (400M)'
        elif model_id.endswith('in21k_ft_in1k'):
            training_data = 'ImageNet-21k (14M)'
        else:
            assert 'in21k' not in model_id
            assert 'clip' not in model_id
        registry.add_model(ModelBase(
            model=model,
            transform_fn=transform,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='transformer',
                arch='vit',
                model_id=f'timm/{model_id}',
                training_data=training_data,
                source='timm',
                eval_batch_size=1,
                extra_annotations=extra_annotations
            )
        ))
