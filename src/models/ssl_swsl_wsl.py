import collections

import torch

from src.datasets import STANDARD_TRANSFORM
from src.models.model_base import ModelBase, ModelMetadata


TorchHubModel = collections.namedtuple('TorchHubModel', ['arch', 'torch_hub_repo'])

model_ids = {
    # === SEMI-WEAKLY SUPERVISED MODELS PRETRAINED WITH 940 HASHTAGGED PUBLIC CONTENT ===
    'resnet18_swsl': TorchHubModel('resnet', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnet50_swsl': TorchHubModel('resnet', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext50_32x4d_swsl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x4d_swsl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x8d_swsl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x16d_swsl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),

    # ================= SEMI-SUPERVISED MODELS PRETRAINED WITH YFCC100M ==================
    'resnet18_ssl': TorchHubModel('resnet', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnet50_ssl': TorchHubModel('resnet', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext50_32x4d_ssl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x4d_ssl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x8d_ssl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x16d_ssl': TorchHubModel('resnext', 'facebookresearch/semi-supervised-ImageNet1K-models'),

    # === WEAKLY-SUPERVISED MODELS PRETRAINED WITH 940 HASHTAGGED PUBLIC CONTENT ===
    'resnext101_32x8d_wsl': TorchHubModel('resnext', 'facebookresearch/WSL-Images'),
    'resnext101_32x16d_wsl': TorchHubModel('resnext', 'facebookresearch/WSL-Images'),
    'resnext101_32x32d_wsl': TorchHubModel('resnext', 'facebookresearch/WSL-Images'),
    'resnext101_32x48d_wsl': TorchHubModel('resnext', 'facebookresearch/WSL-Images'),
}


def add_models(registry):
    for model_id in model_ids:
        torch_hub_model = model_ids[model_id]
        model = torch.hub.load(torch_hub_model.torch_hub_repo, model_id)
        registry.add_model(ModelBase(
            model=model,
            transform_fn=STANDARD_TRANSFORM,
            classify_batch_fn=lambda m, imgs: m(imgs).softmax(-1),
            model_metadata=ModelMetadata(
                arch_family='cnn',
                arch=torch_hub_model.arch,
                model_id=f'facebook/{model_id}',
                training_data='Instagram (940M)' if model_id.endswith('wsl') else 'Flickr YFCC (90M)',
                source='torch.hub',
                eval_batch_size=16 if model_id != 'resnext101_32x48d_wsl' else 4,
                extra_annotations=['Semi-Supervised Learning']
            )
        ))
