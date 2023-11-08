import collections

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

from torchvision.models import alexnet, AlexNet_Weights

from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import densenet161, DenseNet161_Weights
from torchvision.models import densenet169, DenseNet169_Weights
from torchvision.models import densenet201, DenseNet201_Weights

from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights

from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights
from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights

from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import vgg13, VGG13_Weights
from torchvision.models import vgg13_bn, VGG13_BN_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import vgg19_bn, VGG19_BN_Weights

from src.models.model_base import ModelBase, ModelMetadata


TorchvisionModel = collections.namedtuple('TorchvisionModel', ['arch', 'model_cls', 'weights'])
TimmModel = collections.namedtuple('TimmModel', ['arch'])

model_ids = {
    # AlexNet
    'torchvision/alexnet': TorchvisionModel('alexnet', alexnet, AlexNet_Weights),

    # DenseNet
    'torchvision/densenet121': TorchvisionModel('densenet', densenet121, DenseNet121_Weights),
    'torchvision/densenet161': TorchvisionModel('densenet', densenet161, DenseNet161_Weights),
    'torchvision/densenet169': TorchvisionModel('densenet', densenet169, DenseNet169_Weights),
    'torchvision/densenet201': TorchvisionModel('densenet', densenet201, DenseNet201_Weights),

    # GoogLeNet
    'torchvision/googlenet': TorchvisionModel('googlenet', googlenet, GoogLeNet_Weights),

    # Inception_v3
    'torchvision/inception_v3': TorchvisionModel('inception', inception_v3, Inception_V3_Weights),

    # ShuffleNet
    'torchvision/shufflenet_v2_x0_5': TorchvisionModel('shufflenet', shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights),
    'torchvision/shufflenet_v2_x1_0': TorchvisionModel('shufflenet', shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights),
    'torchvision/shufflenet_v2_x1_5': TorchvisionModel('shufflenet', shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights),
    'torchvision/shufflenet_v2_x2_0': TorchvisionModel('shufflenet', shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights),

    # SqueezeNet
    'torchvision/squeezenet1_0': TorchvisionModel('squeezenet', squeezenet1_0, SqueezeNet1_0_Weights),
    'torchvision/squeezenet1_1': TorchvisionModel('squeezenet', squeezenet1_1, SqueezeNet1_1_Weights),

    # VGG
    'torchvision/vgg11': TorchvisionModel('vgg', vgg11, VGG11_Weights),
    'torchvision/vgg11_bn': TorchvisionModel('vgg', vgg11_bn, VGG11_BN_Weights),
    'torchvision/vgg13': TorchvisionModel('vgg', vgg13, VGG13_Weights),
    'torchvision/vgg13_bn': TorchvisionModel('vgg', vgg13_bn, VGG13_BN_Weights),
    'torchvision/vgg16': TorchvisionModel('vgg', vgg16, VGG16_Weights),
    'torchvision/vgg16_bn': TorchvisionModel('vgg', vgg16_bn, VGG16_BN_Weights),
    'torchvision/vgg19': TorchvisionModel('vgg', vgg19, VGG19_Weights),
    'torchvision/vgg19_bn': TorchvisionModel('vgg', vgg19_bn, VGG19_BN_Weights),
}

model_ids_timm = {
    # DenseNet
    'densenet121.ra_in1k': TimmModel('densenet'),
    'densenetblur121d.ra_in1k': TimmModel('densenet'),

    # Inception
    'inception_resnet_v2.tf_ens_adv_in1k': TimmModel('resnet'),  # Adversarial training
    'inception_resnet_v2.tf_in1k': TimmModel('resnet'),
    'inception_v3.gluon_in1k': TimmModel('inception'),
    'inception_v3.tf_adv_in1k': TimmModel('inception'),  # Adversarial training
    'inception_v3.tf_in1k': TimmModel('inception'),
    'inception_v4.tf_in1k': TimmModel('inception'),

    # (Rep)VGG
    'repvgg_a2.rvgg_in1k': TimmModel('vgg'),
    'repvgg_b0.rvgg_in1k': TimmModel('vgg'),
    'repvgg_b1.rvgg_in1k': TimmModel('vgg'),
    'repvgg_b1g4.rvgg_in1k': TimmModel('vgg'),
    'repvgg_b2.rvgg_in1k': TimmModel('vgg'),
    'repvgg_b2g4.rvgg_in1k': TimmModel('vgg'),
    'repvgg_b3.rvgg_in1k': TimmModel('vgg'),
    'repvgg_b3g4.rvgg_in1k': TimmModel('vgg'),
}


def add_models(registry):
    for model_id in model_ids:
        torchvision_model = model_ids[model_id]
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
        # Adversarial Training:
        if model_id in ['inception_v3.tf_adv_in1k', 'inception_resnet_v2.tf_ens_adv_in1k']:
            assert model_id.endswith('adv_in1k')
            extra_annotations = ['Adversarial Training']
        else:
            assert 'adv' not in model_id
        assert 'in21k' not in model_id and 'in22k' not in model_id
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
