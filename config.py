import torch

default_device = torch.device('cuda:0')


class Configurations:
    @staticmethod
    def get_pptbf_fitting_params():
        return PPTBFFittingParams()

    @staticmethod
    def get_noise_model_params(map_name):
        assert map_name in ['albedo', 'height', 'roughness']
        if map_name == 'albedo':
            return NoiseModelParamsConfig0()
        elif map_name == 'roughness':
            return NoiseModelParamsConfig1()
        else:
            return NoiseModelParamsConfig2()

    @staticmethod
    def get_optimization_parameters():
        return OptimizationParams()


class PPTBFFittingParams:
    def __init__(self):
        self.query_lbp = 1e2
        self.query_fft = 1e-2
        self.optim_lbp = 1e2
        self.optim_fft = 1e-2
        self.enable_cropping = False
        self.optimize_closest_query_only = True
        self.num_neighbors = 1
        self.n_step = 5
        self.max_iter_per_continuous_step = 2
        self.max_iter_per_discrete_step = 300  # 200
        self.ratio = 0.8


class NoiseModelParams:
    def __init__(self):
        self.erode_ratio = None
        self.var_threshold = None
        self.n_iters = None
        self.pre_ksize = None
        self.noise_estimators = None
        self.ignore_last = None
        self.last_noise_estimator = None

    def set_params(self):
        pass


class NoiseModelParamsConfig0(NoiseModelParams):
    def __init__(self):
        super().__init__()
        self.set_params()

    def set_params(self):
        self.erode_ratio = 0.01
        self.var_threshold = 0.001
        self.n_iters = 2
        self.pre_ksize = [15, 29, 57, 113]

        self.noise_estimators = [LocalNoise64(),
                                 LocalNoise64(),
                                 LocalNoise128(),
                                 LocalNoise256()]

        assert len(self.pre_ksize) == len(self.noise_estimators)

        self.ignore_last = False
        self.last_noise_estimator = LocalNoiseFullRes()


class NoiseModelParamsConfig1(NoiseModelParams):
    def __init__(self):
        super().__init__()
        self.set_params()

    def set_params(self):
        self.erode_ratio = 0.01
        self.var_threshold = 0.001
        self.n_iters = 2
        self.pre_ksize = [15, 29, 57, 113]

        self.noise_estimators = [LocalNoise64(),
                                 LocalNoise64(),
                                 LocalNoise128(),
                                 LocalNoise256()]

        assert len(self.pre_ksize) == len(self.noise_estimators)

        self.ignore_last = False
        self.last_noise_estimator = ReconNoiseOpenCV(inpaint_radius=0.1, gabor=True)


class NoiseModelParamsConfig2(NoiseModelParams):
    def __init__(self):
        super().__init__()
        self.set_params()

    def set_params(self):
        self.erode_ratio = 0.01
        self.var_threshold = 0.001
        self.n_iters = 2
        self.pre_ksize = [15, 29, 57, 113]

        self.noise_estimators = [ReconNoiseOpenCV(inpaint_radius=0.1, gabor=False),
                                 ReconNoiseOpenCV(inpaint_radius=0.1, gabor=False),
                                 ReconNoiseOpenCV(inpaint_radius=0.1, gabor=False),
                                 ReconNoiseOpenCV(inpaint_radius=0.1, gabor=False)]

        assert len(self.pre_ksize) == len(self.noise_estimators)

        self.ignore_last = False
        self.last_noise_estimator = ReconNoiseOpenCV(inpaint_radius=0.1, gabor=True)


class NoiseSpectrumEstimator:
    def __init__(self):
        self.type = None

    def set_params(self):
        pass

    def print(self):
        pass


class LocalNoise(NoiseSpectrumEstimator):
    def __init__(self):
        super().__init__()
        self.type = 'local'
        self.T = 64
        self.percentage = 0.99
        self.step = 4
        self.min_num_samples = 64

    def print(self):
        print(f'Local PSD: T={self.T}, percentage={self.percentage}, step={self.step}, min # samples={self.min_num_samples}')


class LocalNoise64(LocalNoise):
    def __init__(self):
        super().__init__()
        self.set_params()

    def set_params(self):
        self.T = 64


class LocalNoise128(LocalNoise):
    def __init__(self):
        super().__init__()
        self.set_params()

    def set_params(self):
        self.T = 128


class LocalNoise256(LocalNoise):
    def __init__(self):
        super().__init__()
        self.set_params()

    def set_params(self):
        self.T = 256


class LocalNoiseFullRes(LocalNoise):
    def __init__(self):
        super().__init__()
        self.set_params()

    def set_params(self):
        self.T = 0
        self.percentage = 0.0


class ReconNoise(NoiseSpectrumEstimator):
    def __init__(self):
        super().__init__()
        self.type = 'recon'
        self.method = None
        self.params = {}
        self.gabor = False

    def print(self):
        print(f'Recon PSD: method={self.method}, params={self.params}, gabor={self.gabor}')


class ReconNoiseOpenCV(ReconNoise):
    def __init__(self, inpaint_radius, gabor):
        super().__init__()
        self.method = 'opencv'
        self.params['opencv_inpaint_radius'] = inpaint_radius
        self.gabor = gabor


class ReconNoisePatchMatch(ReconNoise):
    def __init__(self, erode_ratio, gabor):
        super().__init__()
        self.method = 'patchmatch'
        self.params['patchmatch_erode_ratio'] = erode_ratio
        self.gabor = gabor


class OptimizationParams:
    def __init__(self):
        self.lr, self.n_iter = 0.02, 500  # 0.02, 1000
        self.milestones = [500]
        self.vis_every = 50
        self.target_size = None  # (256, 256)
        self.auto_infer = True
        self.display_port = 8000

        # losses
        self.use_l1 = True
        self.match_height = True
        self.n_renders = 4
        self.render_loss_weight = 1
        self.pixel_loss_weight = 0
        self.ssim_loss_weight = 0
        self.vgg_loss_weight = 1

        self.feature_layers = ['relu3_2']  # []
        self.feature_weights = [1]  # []
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        self.style_weights = [1, 1, 1, 1]  # [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
