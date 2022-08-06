import os
import os.path as pth
from copy import deepcopy
from functools import partial

import numpy as np
from GPyOpt.methods import BayesianOptimization
from scipy.optimize import minimize, approx_fprime

import PPTBF.vgg as vgg
from PPTBF.utils import load_parameter_file, save_parameter_file

domain = [
    {'name': 'threshold', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'tiling', 'type': 'discrete', 'domain': tuple(i for i in range(0, 18))},  # 18
    {'name': 'jittering', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'resolution', 'type': 'continuous', 'domain': (1, 1000)},  # 1000
    # {'name': 'resolution', 'type': 'discrete', 'domain': tuple(i for i in range(1, 1001))},  # 1000
    {'name': 'rotation', 'type': 'continuous', 'domain': (0, 2)},
    {'name': 'aspect_ratio', 'type': 'continuous', 'domain': (0.01, 10)},
    {'name': 'distortion_base', 'type': 'continuous', 'domain': (0, 0.25)},
    {'name': 'distortion_amplitude', 'type': 'continuous', 'domain': (0, 4)},
    {'name': 'distortion_frequency', 'type': 'continuous', 'domain': (0, 2)},  # was (0, 1)
    {'name': 'window_shape', 'type': 'discrete', 'domain': tuple(i for i in range(0, 4))},  # 4
    {'name': 'window_arity', 'type': 'continuous', 'domain': (2, 10)},
    {'name': 'window_larp', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'window_norm', 'type': 'continuous', 'domain': (1, 3)},
    {'name': 'window_smoothness', 'type': 'continuous', 'domain': (0, 2)},
    {'name': 'window_blend', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'window_sigma', 'type': 'continuous', 'domain': (0.01, 4)},
    {'name': 'feature_mixture', 'type': 'discrete', 'domain': tuple(i for i in range(0, 5))},  # 5
    {'name': 'feature_norm', 'type': 'continuous', 'domain': (1, 3)},
    {'name': 'feature_correlation', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'feature_aniso', 'type': 'continuous', 'domain': (0, 5)},
    {'name': 'feature_kernel_min', 'type': 'discrete', 'domain': tuple(i for i in range(0, 17))},  # 17
    {'name': 'feature_kernel_max', 'type': 'discrete', 'domain': tuple(i for i in range(0, 17))},  # 17
    {'name': 'feature_sigcos', 'type': 'continuous', 'domain': (0, 10)},
    {'name': 'feature_sigcosvar', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'feature_frequency', 'type': 'discrete', 'domain': tuple(i for i in range(0, 17))},  # 17
    {'name': 'feature_phase_shift', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'feature_thickness', 'type': 'continuous', 'domain': (0.001, 1)},
    {'name': 'feature_curvature', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'feature_orientation', 'type': 'continuous', 'domain': (0, 0.5)},
]

important_param_names = ['tiling', 'resolution', 'window_shape', 'feature_mixture', 'feature_frequency']


def decode(params):
    # continuous params
    params_continuous = []
    lb = []
    ub = []
    # important discrete params
    params_discrete_imp = []
    enum_imp = []
    domain_discrete_imp = []
    # inferior discrete params
    params_discrete_inf = []
    enum_inf = []
    domain_discrete_inf = []
    reverse_mapping = []
    for id, (desc, param) in enumerate(zip(domain, params)):
        param_name = desc['name']
        param_type = desc['type']
        param_domain = desc['domain']

        if param_type == 'continuous':
            params_continuous.append(param)
            lb.append(param_domain[0])
            ub.append(param_domain[1])
            reverse_mapping.append(('continuous', len(params_continuous) - 1))
        elif param_type == 'discrete':
            if param_name in important_param_names:
                params_discrete_imp.append(param)
                enum_imp.append(param_domain)
                domain_discrete_imp.append(desc)
                reverse_mapping.append(('discrete_imp', len(params_discrete_imp) - 1))
            else:
                params_discrete_inf.append(param)
                enum_inf.append(param_domain)
                domain_discrete_inf.append(desc)
                reverse_mapping.append(('discrete_inf', len(params_discrete_inf) - 1))
        else:
            raise RuntimeError("Unknown parameter type")

    return {'params_continuous': params_continuous, 'lb': lb, 'ub': ub,
            'params_discrete_imp': params_discrete_imp, 'enum_imp': enum_imp, 'domain_discrete_imp': domain_discrete_imp,
            'params_discrete_inf': params_discrete_inf, 'enum_inf': enum_inf, 'domain_discrete_inf': domain_discrete_inf,
            'reverse_mapping': reverse_mapping}


class Optimizer(object):
    def __init__(self, initial_params, target_image, weight_fft, weight_lbp, output_path, filename, **kwargs):
        self.initial_params = initial_params
        self.target_image = target_image
        self.weight_fft = weight_fft
        self.weight_lbp = weight_lbp

        self.generator = kwargs['generator']
        self.enable_cropping = kwargs['enable_cropping']

        params_dict = decode(initial_params)
        # continuous params
        self.params_continuous = np.asarray(params_dict['params_continuous'])
        self.lb = np.asarray(params_dict['lb'])
        self.ub = np.asarray(params_dict['ub'])
        # discrete params
        self.params_discrete = {'imp': np.asarray(params_dict['params_discrete_imp']),
                                'inf': np.asarray(params_dict['params_discrete_inf'])}
        self.enum = {'imp': params_dict['enum_imp'], 'inf': params_dict['enum_inf']}
        self.domain_discrete = {'imp': params_dict['domain_discrete_imp'], 'inf': params_dict['domain_discrete_inf']}
        constraint_inf = [{'name': 'constraint0', 'constraint': 'x[:,0]-x[:,1]'}]
        self.constraint_discrete = {'imp': None, 'inf': constraint_inf}
        # reverse mapping for assembling
        self.reverse_mapping = params_dict['reverse_mapping']

        n_params = len(self.lb)
        self.bounds = [(0, 1) for _ in range(n_params)]
        self.normalizer = Normalizer(self.lb, self.ub)
        self.params_continuous = self.normalizer.normalize(self.params_continuous)

        # define outputs
        self.output_path = output_path
        self.temp_image_filename = pth.join(output_path, filename + '_temp_pptbf_binary.png')
        self.output_params_filename = os.path.join(output_path, filename + '_final_pptbf_params.txt')
        self.global_params_filename = os.path.join(output_path, filename + '_global_pptbf_params.txt')

        self.device = kwargs['device']
        self.vgg19 = kwargs['vgg']

        # define states of optimization
        self.flag = None
        self.fixed_params = None
        self.method = 'Powell'  # Powell # L-BFGS-B # SLSQP #TNC
        self.discrete_method = 'Bayesian'  # Only support Bayesian
        self.global_min = np.inf
        self.global_optimal_continuous = self.params_continuous.copy()
        self.global_optimal_discrete = deepcopy(self.params_discrete)

        # load masks if exists
        self.mask = kwargs['mask']
        if self.mask is not None:
            self.target_image *= self.mask

    def assemble(self, params_continuous, params_discrete):
        def assemble_(dict_params):
            params = []
            for mapping in self.reverse_mapping:
                params.append(dict_params[mapping[0]][mapping[1]])
            return params

        params_continuous_ = self.normalizer.denormalize(params_continuous)  # denormalize
        return assemble_({'continuous': params_continuous_, 'discrete_imp': params_discrete['imp'],
                          'discrete_inf': params_discrete['inf']})

    def evaluate(self, parameter):
        if self.enable_cropping and self.flag is not None:
            self.generator.generate(width=1024, height=1024, parameters=parameter, save_result=True)
            image = vgg.image_loader(self.temp_image_filename, crop_size=512).to(self.device)
        else:
            self.generator.generate(width=512, height=512, parameters=parameter, save_result=True)
            image = vgg.image_loader(self.temp_image_filename).to(self.device)
        return image

    def loss_func(self, params):
        if self.flag == 'c':  # optimize continuous one
            full_params = self.assemble(params_continuous=params, params_discrete=self.fixed_params)
        elif self.flag == 'imp':  # optimize discrete one (important)
            full_params = self.assemble(params_continuous=self.fixed_params['c'],
                                        params_discrete={'imp': params[0], 'inf': self.fixed_params['inf']})
        elif self.flag == 'inf':  # optimize discrete one (inferior)
            full_params = self.assemble(params_continuous=self.fixed_params['c'],
                                        params_discrete={'imp': self.fixed_params['imp'], 'inf': params[0]})
        elif self.flag is None:
            full_params = self.assemble(params_continuous=self.params_continuous, params_discrete=self.params_discrete)
        else:
            raise RuntimeError("Unknown flag")

        fake_image = self.evaluate(full_params)
        if self.mask is not None:
            fake_image *= self.mask

        l1, l2, l3 = vgg.loss(source=fake_image, target=self.target_image, vgg=self.vgg19,
                              weight_fft=self.weight_fft, weight_lbp=self.weight_lbp)

        style_loss = l1.cpu().numpy()
        fft_loss = l2.cpu().numpy()
        lbp_loss = l3

        loss = style_loss + fft_loss + lbp_loss

        # if a global minimum loss is sampled during loss evaluation, record this set of parameters
        if params is not None and loss < self.global_min:
            if check_valid(full_params):
                print(f'Global minimum loss = {loss}')
                self.global_min = loss
                if self.flag == 'c':
                    self.global_optimal_continuous = params.copy()
                else:
                    self.global_optimal_discrete[self.flag] = params[0].copy()

                global_optimal = self.assemble(self.global_optimal_continuous, self.global_optimal_discrete)
                save_parameter_file(global_optimal, self.global_params_filename)
                self.generator.generate(width=512, height=512, filename=self.global_params_filename)

        return loss

    def optimize_continuous_params(self, max_iter):
        self.flag = 'c'
        self.fixed_params = {'imp': self.params_discrete['imp'].copy(), 'inf': self.params_discrete['inf'].copy()}

        gradient = partial(approx_fprime, f=self.loss_func, epsilon=0.001)
        res = minimize(self.loss_func, self.params_continuous.copy(), method=self.method, jac=gradient,
                       bounds=self.bounds,
                       options={"maxiter": max_iter, 'disp': False}, tol=1e-4)

        self.params_continuous = self.global_optimal_continuous.copy()

    def optimize_discrete_params(self, max_iter, flag):
        assert (flag in ['imp', 'inf'])

        self.flag = flag
        another = 'imp' if flag == 'inf' else 'inf'

        self.fixed_params = {'c': self.params_continuous.copy(), another: self.params_discrete[another].copy()}
        domain_discrete = self.domain_discrete[flag]
        constraints = self.constraint_discrete[flag]

        X = np.asarray([self.params_discrete[flag].copy()])
        Y = np.asarray([[self.loss_func(X)]])

        optimizer = BayesianOptimization(f=self.loss_func, domain=domain_discrete, constraints=constraints, initial_design_numdata=0)
        optimizer.X = X
        optimizer.Y = Y

        optimizer.run_optimization(max_iter=max_iter)

        self.params_discrete[flag] = self.global_optimal_discrete[flag].copy()

    def optimize(self, n_steps, max_iter_per_continuous_step, max_iter_per_discrete_step, ratio, timer):
        max_iter_per_discrete_step_imp = int(max_iter_per_discrete_step * ratio)
        max_iter_per_discrete_step_inf = max_iter_per_discrete_step - max_iter_per_discrete_step_imp

        self.flag = None
        print(f'initial loss = {self.loss_func(None)}')

        # alternative optimization
        timer.begin()

        self.global_min = np.inf
        for i_step in range(n_steps):
            timer.begin()
            self.optimize_continuous_params(max_iter=max_iter_per_continuous_step)
            timer.end(f'Continuous Optimization {i_step} complete in ')

            timer.begin()
            self.optimize_discrete_params(max_iter=max_iter_per_discrete_step_imp, flag='imp')
            timer.end(f'Discrete (Important) Optimization {i_step} complete in ')

            timer.begin()
            self.optimize_discrete_params(max_iter=max_iter_per_discrete_step_inf, flag='inf')
            timer.end(f'Discrete (Inferior) Optimization {i_step} complete in ')

        timer.begin()
        self.optimize_continuous_params(max_iter=max_iter_per_continuous_step)
        timer.end('Final round Continuous Optimization complete in ')

        timer.end('Full Optimization complete in ')

        self.flag = None
        final_loss = self.loss_func(None)
        print(f'Final loss = {final_loss}')

        optimal_params = self.assemble(self.params_continuous, self.params_discrete)
        save_parameter_file(optimal_params, self.output_params_filename)
        self.generator.generate(width=512, height=512, filename=self.output_params_filename)

        return final_loss


class Normalizer(object):
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.range = ub - lb

    def normalize(self, x):
        return (x - self.lb) / self.range

    def denormalize(self, x):
        return x * self.range + self.lb


def check_valid(params):
    err = False
    for param, desc in zip(params, domain):
        if desc['type'] == 'continuous':
            if not (desc['domain'][0] <= param <= desc['domain'][1]):
                err = True
                print(f"{desc['name']}: {param}; {desc['domain']}")

        elif desc['type'] == 'discrete':
            if param not in desc['domain']:
                err = True
                print(f"{desc['name']}: {param}; {desc['domain']}")
        else:
            raise RuntimeError
    return not err


def print_params(file_path):
    def visualize(params):
        for p, desc in zip(params, domain):
            print(f"{desc['name']}({desc['type']}): {p:.3f}")
    params = load_parameter_file(file_path)[0]
    visualize(params)