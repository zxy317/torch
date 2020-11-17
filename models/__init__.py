# -*-coding:utf-8-*-
from .alexnet import *
from .cbam_resnext import *
from .densenet import *
from .genet import *
from .lenet import *
from .preresnet import *
from .resnet import *
from .resnext import *
from .senet import *
from .shake_shake import *
from .sknet import *
from .vgg import *


def get_model(name_model, num_classes):
    if name_model == "lenet":
        return lenet(num_classes)
    elif name_model == "vgg":
        return vgg16(num_classes)
    elif name_model == "alexnet":
        return alexnet(num_classes)
    elif name_model == "cbam_resnext":
        return cbam_resnext29_8x64d(num_classes)
    elif name_model == "densenet":
        return densenet100bc(num_classes)
    elif name_model == "genet":
        return ge_resnext29_8x64d(num_classes)
    elif name_model == "preresnet":
        return preresnet20(num_classes)
    elif name_model == "resnet":
        return resnet32(num_classes)
    elif name_model == "resnext":
        return resnext29_8x64d(num_classes)
    elif name_model == "senet":
        return se_resnext29_8x64d(num_classes)
    elif name_model == "shake_shake":
        return shake_resnet26_2x32d(num_classes)
    elif name_model == "sknet":
        return sk_resnext29_16x32d(num_classes)