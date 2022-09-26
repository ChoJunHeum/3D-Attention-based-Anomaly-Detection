import os
from os.path import join


class Config(object):
    def __init__(self, args=None):
        self.tr_name        = 'shanghai_cropped_08'
        self.te_name        = 'shanghaitech'
        self.device         = 'cuda'

        # pathes
        self.cwd            = os.getcwd()
        self.ad_home        = os.path.dirname(self.cwd)
        self.data_home      = join(self.ad_home, 'm8/datasets')
        # self.result_name    = 'results'
        self.save_name      = 'checkpoints'

        # images
        self.nframe         = 4
        self.resize_h       = 64
        self.resize_w       = 64
        self.max_h          = 480
        self.max_w          = 856
        self.symmetric      = False
        self.factor_x       = 1.5
        self.factor_y       = 1.2
        self.confidence     = 0.4
        self.gamma          = 0.3
        
        # model
        self.generator      = 'custom4'
        self.n_kernels      = [32, 64, 64, 128] # [32,64,64,128]
        self.bn_kernel      = 128
        self.flownet        = 'lite'
        self.att_dim        = 256

        # train
        self.resume         = None
        self.batch_size     = 32
        self.epoch          = 10
        self.save_epoch     = 5
        self.verbose        = 10
        self.g_lr           = 1e-3
        self.d_lr           = 1e-4
        self.l2             = 1e-5 # weight decay for optimizer (1e-5 or 1e-2)
        self.init           = 'kaiming'
        self.optimizer      = 'adamw'
        self.scheduler      = 'cosine'
        self.warm_cycle     = 10000 # used only if scheduler='cosinewr'
        self.lambdas        = [1., 1., 1., 2.] # coef for intensity, gradient, adverserial, flownet loss
                                # original [1., 1., .05, 2.]
        # test
        # self.show_curve     = True
        # self.trained_model  = None

        self._update_config(args)
        self._check_path()

    def _update_config(self, args=None):
        if args is not None:
            for k, v in vars(args).items():
                self.__setattr__(k, v)
        if self.generator == 'custom3':
            assert len(self.n_kernels) == 3

    def _check_path(self):
        # self.result_prefix = join(self.cwd, self.result_name)
        self.save_prefix = join(self.cwd, self.save_name)
        self.tr_path = join(self.data_home, self.tr_name)
        self.te_path = join('/home/chojh21c/ADGW/Future_Frame_Prediction/datasets', self.te_name)
        for path in [self.tr_path, self.te_path]:
            assert os.path.exists(path), 'Check dataset path'

        # for path in [self.result_prefix, self.save_prefix]:
        for path in [self.save_prefix]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def desc_cfg(self):
        desc = '=======Configurations=======\n'
        for k, v in vars(self).items():
            desc += f'{k}: {v} \n'
        return desc