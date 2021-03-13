import torch
from torch import nn

from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet
import pretrainedmodels
from pretrainedmodels import utils

def generate_C2D_model(opt):
    if opt.c2d_model_name == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif opt.c2d_model_name == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif opt.c2d_model_name == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif opt.c2d_model_name == 'inceptionresnetv2':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    model.last_linear = utils.Identity()

    if not opt.no_cuda:
        model = model.to(opt.device)

    return load_image_fn, model, (C, H, W)


def generate_C3D_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.c3d_model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

    if opt.c3d_model_name == 'resnet':
        assert opt.c3d_model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.c3d_model_depth == 10:
            model = resnet.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.c3d_model_depth == 18:
            model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.c3d_model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.c3d_model_depth == 50:
            model = resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.c3d_model_depth == 101:
            model = resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.c3d_model_depth == 152:
            model = resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.c3d_model_depth == 200:
            model = resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
    elif opt.c3d_model_name == 'wideresnet':
        assert opt.c3d_model_depth in [50]

        if opt.c3d_model_depth == 50:
            model = wide_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, k=opt.wide_resnet_k,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
    elif opt.c3d_model_name == 'resnext':
        assert opt.c3d_model_depth in [50, 101, 152]

        if opt.c3d_model_depth == 50:
            model = resnext.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.c3d_model_depth == 101:
            model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
        elif opt.c3d_model_depth == 152:
            model = resnext.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
    elif opt.c3d_model_name == 'preresnet':
        assert opt.c3d_model_depth in [18, 34, 50, 101, 152, 200]

        if opt.c3d_model_depth == 18:
            model = pre_act_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.c3d_model_depth == 34:
            model = pre_act_resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.c3d_model_depth == 50:
            model = pre_act_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.c3d_model_depth == 101:
            model = pre_act_resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.c3d_model_depth == 152:
            model = pre_act_resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.c3d_model_depth == 200:
            model = pre_act_resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
    elif opt.c3d_model_name == 'densenet':
        assert opt.c3d_model_depth in [121, 169, 201, 264]

        if opt.c3d_model_depth == 121:
            model = densenet.densenet121(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.c3d_model_depth == 169:
            model = densenet.densenet169(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.c3d_model_depth == 201:
            model = densenet.densenet201(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.c3d_model_depth == 264:
            model = densenet.densenet264(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)

    # print(model)
    print('loading c3d model from: {}'.format(opt.c3d_model_checkpoint))
    model_data = torch.load(opt.c3d_model_checkpoint)
    print(model_data['arch'])
    assert opt.arch == model_data['arch']

    model_state_dict = {}
    for k, v in model_data['state_dict'].items():
        model_state_dict[k[k.index('.') + 1:]] = v

    model.load_state_dict(model_state_dict)

    if not opt.no_cuda:
        model = model.to(opt.device)
        # model = nn.DataParallel(model, device_ids=None)

    # print(model)
    return model
