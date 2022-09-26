import glob
import os
import argparse

from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',     '-ck', default=None, type=str, required=True)
parser.add_argument('--generator',      '-g', default='custom4', type=str,
                                            choices=['vgg16unet', 'vgg13unet', 'custom3', 'custom4'])
parser.add_argument('--n_kernels',      '-nk', default=[32,64,64,128], nargs='+', type=int)
parser.add_argument('--bn_kernel',      '-bk', default=128, type=int)      
parser.add_argument('--att_dim',        '-att', default=512, type=int)                                      
parser.add_argument('--device',         '-device', default='cuda', type=str)
parser.add_argument('--gpu_num',        '-gpu', default='0', type=str)
parser.add_argument('--nframe',         '-nf', default=4, type=int)
parser.add_argument('--resize_h',       '-rsh', default=64, type=int)
parser.add_argument('--resize_w',       '-rsw', default=64, type=int)
parser.add_argument('--factor_x',       '-fx', default=1.5, type=float)
parser.add_argument('--factor_y',       '-fy', default=1.2, type=float)
parser.add_argument('--max_h',          '-mxh', default=480, type=int)
parser.add_argument('--max_w',          '-mxw', default=856, type=int)
parser.add_argument('--confidence',     '-c', default=.4, type=float)
parser.add_argument('--gamma',          '-gamma', default=.3, type=float)


def main(cfg):
    import torch

    import init_utils
    import data_utils
    import ops

    torch.multiprocessing.set_start_method('spawn')
    yolo_model = init_utils.get_yolo(cfg)

    test_dirs = glob.glob(f'{cfg.te_path}/testing/*_*')
    test_dirs.sort()

    # Label_loader: list[array[0~1]*n]

    label_loader = data_utils.Label_loader(cfg, test_dirs)
    labels = label_loader()

    transform = data_utils.get_transform(cfg)
    print(f'yolo_model.training: {yolo_model.training}')
    
    print(f'start ckpt: {cfg.checkpoint}')
    generator = init_utils.get_ckpt_generator(cfg)

    res = ops.infer(generator, yolo_model,
                    test_dirs, labels,
                    transform, cfg, smooth=True)

    log = open(f"results/test.txt", "a")

    print(f'Env: {cfg.checkpoint} / {cfg.confidence} / {cfg.gamma} / {cfg.att_dim} / {cfg.factor_x}, {cfg.factor_y} | auc: {res["auc"]:.4f} | elapsed: {res["elapsed"]/60:.2f} mins')

    log.write(f'Env: {cfg.checkpoint} / {cfg.confidence} / {cfg.gamma} / {cfg.att_dim} / {cfg.factor_x}, {cfg.factor_y} | auc: {res["auc"]:.4f} | elapsed: {res["elapsed"]/60:.2f} mins \n')
    log.close()

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    

    cfg = Config(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_num)

    main(cfg)
    