from os.path import join
import torch
import torch.nn as nn

from modules.custom import Generator3, Generator4
from modules.pix2pix_networks import PixelDiscriminator
from modules.flownet2.models import FlowNet2SD
from modules.liteFlownet import lite_flownet as lite_flow

import losses


def get_models_opts(cfg, N=10000):
    '''
    Return:
        models = [generator, discriminator, flownet]
        opts = [opt_g, opt_d]
        schs = [sch_g, sch_d]
    '''
    gen_name = cfg.generator
    fn_name = cfg.flownet
    resume = cfg.resume
    optimizer_name = cfg.optimizer
    scheduler_name = cfg.scheduler
    in_channels = 3
    # in_channels = cfg.nframe * 3
    out_channels = 3
    info = {}

    # generator and discriminator
    if gen_name == 'custom4':
        generator = Generator4(in_channels, out_channels, cfg=cfg)
    elif gen_name == 'custom3':
        generator = Generator3(in_channels, out_channels, cfg=cfg)        
    else:
        raise NotImplementedError
    discriminator = PixelDiscriminator(input_nc=out_channels)
    
    # flownet
    if fn_name == 'lite':
        flownet = lite_flow.Network()
        flownet.load_state_dict(
            torch.load(join(cfg.ad_home, 'm8/pretrained/network-default.pytorch')))
    elif fn_name == '2sd':
        flownet = FlowNet2SD()
        flownet.load_state_dict(
            torch.load(join(cfg.ad_home, 'm8/pretrained/FlowNet2-SD.pth'))['state_dict'])
    else:
        raise NotImplementedError

    models = [generator, discriminator, flownet]
    _ = [m.to(cfg.device) for m in models]

    # optimizer and scheduler
    if optimizer_name == 'adamw':
        opt_g = torch.optim.AdamW(
                    generator.parameters(),
                    lr=cfg.g_lr,
                    weight_decay=cfg.l2,
                    )
        opt_d = torch.optim.AdamW(
                    discriminator.parameters(),
                    lr=cfg.d_lr,
                    weight_decay=cfg.l2,
                    )
    elif optimizer_name == 'adam':
        opt_g = torch.optim.Adam(
                    generator.parameters(),
                    lr=cfg.g_lr,
                    weight_decay=cfg.l2,
                    )
        opt_d = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=cfg.d_lr,
                    weight_decay=cfg.l2,
                    )
    else:
        raise NotImplementedError

    if scheduler_name == 'no':
        sch_g = None; sch_d = None
    elif scheduler_name == 'cosine':
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=opt_g,
                    T_max=int(cfg.epoch*N/cfg.batch_size),
                    eta_min=0,
                    )
        sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=opt_d,
                    T_max=int(cfg.epoch*N/cfg.batch_size),
                    eta_min=0,
                    )
    elif scheduler_name == 'cosinewr':
        sch_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=opt_g,
                    T_0=cfg.warm_cycle,
                    T_mult=2,
                    eta_min=0,
                    )
        sch_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=opt_d,
                    T_0=cfg.warm_cycle,
                    T_mult=2,
                    eta_min=0,
                    )
    else:
        raise NotImplementedError
    
    if resume is not None:
        ckpt = torch.load(cfg.resume, map_location=cfg.device)
        # ckpt = torch.load(cfg.resume)
        generator.load_state_dict(ckpt['net_g'])
        discriminator.load_state_dict(ckpt['net_d'])
        opt_g.load_state_dict(ckpt['opt_g'])
        opt_d.load_state_dict(ckpt['opt_d'])
        if sch_g is not None:
            if 'sch_g' not in list(ckpt.keys()):
                raise Exception('Config and checkpoint do not match')
            sch_g.load_state_dict(ckpt['sch_g'])
            sch_d.load_state_dict(ckpt['sch_d'])
            info['last_lr'] = ckpt['sch_g']['_last_lr'][0]
            info['step'] = ckpt['sch_g']['_step_count']
        else:
            if 'sch_g' in list(ckpt.keys()):
                raise Exception('Config and checkpoint do not match')
        print(f'Pre-trained models and opts have been loaded.')

    else:
        if cfg.init == 'original':
            weights_init = weights_init_original
        elif cfg.init == 'xavier':
            weights_init = weights_init_xavier
        elif cfg.init == 'kaiming':
            weights_init = weights_init_kaiming
        else:
            weights_init = weights_init_normal
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        print('Generator and discriminator are going to be trained from scratch.')

    opts = [opt_g, opt_d]
    schs = [sch_g, sch_d]
    
    return models, opts, schs, info


def get_losses(cfg):
    '''
    Return:
        losses = [adversarial_loss,
                    discriminate_loss,
                    gradient_loss,
                    intensity_loss,
                    flow_loss]
    '''
    adversarial_loss = losses.Adversarial_Loss()
    discriminate_loss = losses.Discriminate_Loss()
    gradient_loss = losses.Gradient_Loss(channels=3)
    intensity_loss = losses.Intensity_Loss()
    flow_loss = losses.Flow_Loss()

    total_loss = [adversarial_loss,
              discriminate_loss,
              gradient_loss,
              intensity_loss,
              flow_loss]

    _ = [l.to(cfg.device) for l in total_loss]

    return total_loss


def weights_init_original(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def save_all(models, opts, schs, save_path):
    generator, discriminator, _ = models
    opt_g, opt_d = opts
    sch_g, sch_d = schs
    
    model_dict = {
            'net_g': generator.state_dict(),
            'net_d': discriminator.state_dict(),
            'opt_g': opt_g.state_dict(),
            'opt_d': opt_d.state_dict(),
            }
            
    if schs[0] is not None:
        model_dict['sch_g'] = sch_g.state_dict()
        model_dict['sch_d'] = sch_d.state_dict()

    torch.save(model_dict, save_path)
    return None


def get_checkpoint_model(cfg):
    save_path = join(cfg.save_prefix, cfg.checkpoint)
    gen_name = cfg.generator
    in_channels = 3    
    out_channels = 3
    yolo_device = f'{cfg.device}:{cfg.gpu_num}'
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, device=yolo_device)
    yolo_model.eval()

    # generator
    if gen_name == 'custom4':
        generator = Generator4(in_channels, out_channels, cfg=cfg)
    elif gen_name == 'custom3':
        generator = Generator3(in_channels, out_channels, cfg=cfg)        
    else:
        raise NotImplementedError
    
    generator.eval()
    generator.to(cfg.device)
    generator.load_state_dict(torch.load(save_path,
                                map_location=cfg.device)['net_g'])
    
    return yolo_model, generator


def get_ckpt_generator(cfg):
    save_path = join(cfg.save_prefix, cfg.checkpoint)
    gen_name = cfg.generator
    in_channels = 3    
    out_channels = 3

    # generator
    if gen_name == 'custom4':
        generator = Generator4(in_channels, out_channels, cfg=cfg)
    elif gen_name == 'custom3':
        generator = Generator3(in_channels, out_channels, cfg=cfg)        
    else:
        raise NotImplementedError
    
    generator.eval()
    generator.to(cfg.device)
    generator.load_state_dict(torch.load(save_path,
                                map_location=cfg.device)['net_g'])
    return generator


def get_yolo(cfg):
    yolo_device = f'{cfg.device}:{cfg.gpu_num}'
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, device=yolo_device)
    return yolo_model.eval()