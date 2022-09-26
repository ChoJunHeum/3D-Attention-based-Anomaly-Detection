import os
import argparse
import logging
from os.path import join

# from torch.utils.tensorboard import SummaryWriter

from config import Config
import log_utils


parser = argparse.ArgumentParser(description='Advision')
parser.add_argument('--tr_name',        '-d', default='shanghai_cropped_08', type=str)
parser.add_argument('--batch_size',     '-bs', default=64, type=int)
parser.add_argument('--generator',      '-g', default='custom4', type=str,
                                            choices=['vgg16unet', 'vgg13unet', 'custom3', 'custom4'])
parser.add_argument('--n_kernels',      '-nk', default=[32, 64, 64, 128], nargs='+', type=int)
parser.add_argument('--bn_kernel',      '-bk', default=128, type=int)
parser.add_argument('--att_dim',        '-att', default=512, type=int)
parser.add_argument('--flownet',        '-fn', default='lite', type=str,
                                            choices=['lite', '2sd'])
parser.add_argument('--device',         '-device', default='cuda', type=str)
parser.add_argument('--gpu_num',        '-gpu', default='0', type=str)
parser.add_argument('--nframe',         '-nf', default=4, type=int)
parser.add_argument('--resize_h',       '-rsh', default=64, type=int)
parser.add_argument('--resize_w',       '-rsw', default=64, type=int)
parser.add_argument('--epoch',          '-e', default=80, type=int) # 10 epoch ~ 5 hours
parser.add_argument('--resume',         '-r', default=None, type=str)
parser.add_argument('--save_epoch',     '-se', default=1, type=int)
parser.add_argument('--verbose',        '-vb', default=100, type=int)
parser.add_argument('--warm_cycle',     '-wc', default=5000, type=int)
parser.add_argument('--init',           '-init', default='kaiming', type=str,
                                            choices=['original', 'xavier', 'normal', 'kaiming'])
parser.add_argument('--optimizer',      '-opt', default='adamw', type=str,
                                            choices=['adam', 'adamw'])
parser.add_argument('--scheduler',      '-sch', default='cosine', type=str,
                                            choices=['cosine', 'no', 'cosinewr'])


def main(logger, log_dir, cfg):
    import torch
    import init_utils
    import data_utils
    import ops 
    import calc_scores

    torch.multiprocessing.set_start_method('spawn', force=True)
    train_dataset = data_utils.CropTrainDataset(cfg)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=cfg.batch_size,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    drop_last=False)
    N = train_dataset.__len__()
    logger.info(f'Dataset len: {N}')
    # init models, optimizers, schedulers, losses
    # all of these are lists. 
    # Check the order in the description for each function.
    models, opts, schs, info = init_utils.get_models_opts(cfg=cfg, N=N)
    losses = init_utils.get_losses(cfg=cfg)
    models[-1].eval() # flow_net.eval()
    _ = [m.train() for m in models[:-1]] # others train()

    if cfg.resume is not None:
        logger.info(f'Train resumed from {cfg.resume}')
        logger.info(f"Last lr: {info['last_lr']}")
        logger.info(f"Step count: {info['step']}")
        step = info['step']

    save_prefix = log_utils.make_date_dir(cfg.save_prefix)
    # save_path = join(save_prefix, 'total_model.pt')
    logger.info(f'Model save path: {save_prefix}')

    start_iter = step if cfg.resume else 0
    best_psnr = 30; best_info=f''
    best_psnr_path = join(save_prefix, "best_total_model.pt")
    save_epochs = [5, 15]
    # valid_epoch = 9 if cfg.batch_size == 32 else 19
    
    try:
        epoch = 0
        global_step = start_iter + 1
        logger.info('Start training')
        for epoch_ in range(cfg.epoch):
            epoch = epoch_ + 1
            for i, batches in enumerate(train_dataloader):
                inputs, targets = batches
                preds = ops.step_train(inputs, targets, models, losses,
                                    opts, schs, cfg,
                                    epoch=epoch, global_step=global_step)

                if global_step % cfg.verbose == 0:
                    preds = preds * data_utils.STD4 + data_utils.MEAN4
                    targets = targets * data_utils.STD4 + data_utils.MEAN4
                    psnr_batch = calc_scores.psnr_error(preds, targets)
                    logger.info(f'[Train-{epoch}-{global_step}] ## Train-PSNR ##: {psnr_batch:.2f}')

                    if psnr_batch > best_psnr:
                        best_psnr = psnr_batch
                        best_info = f'{epoch}-{global_step}'
                        init_utils.save_all(models, opts, schs,
                                            save_path=best_psnr_path)
                        logger.info('################################################')
                        logger.info(f'Update BEST PSNR Score!')
                        logger.info(f'PSNR: {best_psnr:.2f}, Epoch-iter: {best_info}')
                        logger.info('################################################')
                global_step += 1

            if epoch % cfg.save_epoch == 0 or epoch in save_epochs:
                save_path = join(save_prefix, f'total_model_e{epoch}.pt')
                init_utils.save_all(models, opts, schs,
                                    save_path=save_path)
                logger.info(f'[Train-{epoch}-{global_step}] Model_dict saved: {save_path}')

        save_path = join(save_prefix, f'total_model_e{epoch}.pt')                                
        init_utils.save_all(models, opts, schs,
                            save_path=save_path)
        logger.info(f'[Finish-{epoch}-{global_step}] Model_dict saved: {save_path}')
        logger.info(f'[BEST] PSNR: {best_psnr:.2f}, Epoch-iter: {best_info}')
        logger.info('Log path: {}'.format(log_dir))

    except:
        save_path = join(save_prefix, f'total_model_ex.pt')   
        logger.info('Exception Occurred')
        init_utils.save_all(models, opts, schs,
                            save_path=save_path)
        logger.info(f'[Interrupted] Model_dict saved: {save_path}')
        logger.info(f'[BEST] PSNR: {best_psnr:.2f}, Epoch-iter: {best_info}')
        logger.info('Log path: {}'.format(log_dir))
        logging.exception("Exception message:")

    finally:
        logger.handlers.clear()
        logging.shutdown()  

    return None


if __name__ == "__main__":

    args = parser.parse_args()
    train_cfg = Config(args)
    cfg_desc = train_cfg.desc_cfg()

    logger, log_dir = log_utils.get_logger('logs/')

    os.environ["CUDA_VISIBLE_DEVICES"]=str(train_cfg.gpu_num)
    logger.info(f'CUDA_VISIBLE_DEVICES: {train_cfg.gpu_num}')

    logger.info(cfg_desc)
    
    main(logger, log_dir, train_cfg)