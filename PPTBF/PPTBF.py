import glob
import os
import os.path as pth
import pickle
import numpy as np
import torch

from utils import Timer
import PPTBF.PPTBFModel as PPTBFModel
import PPTBF.vgg as vgg
from PPTBF.optimizer import Optimizer
from PPTBF.query import build_nearest_neighbor_arch, query
from PPTBF.utils import save_image, save_parameter_file, load_parameter_file


database = './PPTBF/database'


class PPTBFFitter:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device('cuda')

        # load pre-trained VGG
        self.vgg19 = vgg.VGG19().to(self.device)

        # build searching architecture
        self.flann, self.flann_params = build_nearest_neighbor_arch(root=database, distance_type='euclidean', load_index=True)

        # load the trained IPCA model
        with open(os.path.join(database, 'ipca.pkl'), 'rb') as f:
            self.ipca = pickle.load(f)

        # load the image index file
        with open(os.path.join(database, 'index.txt'), 'r') as f:
            image_index = f.read().replace('\n', ' ').replace('\r', ' ')
            image_index = [index for index in image_index.split(' ') if index]
            self.image_index = [int(index) for index in image_index]

        # list all subdirectories in the root directory
        self.sub_dirs = glob.glob(os.path.join(database, '*/'))

    def fit(self, target_files, mask_files=None):
        if mask_files is None:
            mask_files = [None]*len(target_files)
        elif isinstance(mask_files, str):
            mask_files = [mask_files] * len(target_files)
        else:
            raise RuntimeError('mask_files should be None or a file path')

        optimizers = []

        for target_file, mask_file in zip(target_files, mask_files):
            filename = pth.splitext(pth.basename(target_file))[0]
            output_path = pth.join(pth.dirname(target_file), filename)

            os.makedirs(output_path, exist_ok=True)
            generator = PPTBFModel.PPTBFGenerator()
            generator.output_path = output_path
            generator.filename = filename

            # load the input image and its mask
            target = vgg.image_loader(target_file).to(self.device)
            save_image(target.squeeze(), pth.join(output_path, f'{filename}.png'))
            if mask_file is not None:
                mask = vgg.image_loader(mask_file).to(self.device)
                save_image(mask.squeeze(), os.path.join(output_path, 'prev_bmask.png'))
            else:
                mask = None

            kwargs = {
                'filename': filename,
                'output_path': output_path,
                'vgg': self.vgg19,
                'weight_lbp': self.cfg.query_lbp,
                'weight_fft': self.cfg.query_fft,
                'ipca': self.ipca,
                'flann': self.flann,
                'flann_params': self.flann_params,
                'image_index': self.image_index,
                'sub_dirs': self.sub_dirs,
                'enable_cropping': True
            }

            try:
                initial_parameter = load_parameter_file(pth.join(output_path, filename + '_final_pptbf_params.txt'))[0]
                print('Find previous optimization results. Restart optimization.')
            except FileNotFoundError:
                print("No previous optimization results.")
                initial_parameter, all_parameters = query(image=target, num_neighbors=self.cfg.num_neighbors, save_image=False, **kwargs)

                sub_output_path = pth.join(output_path, 'query')
                os.makedirs(sub_output_path, exist_ok=True)
                for i in range(self.cfg.num_neighbors):
                    params = all_parameters[i]
                    save_parameter_file(params, pth.join(sub_output_path, f'{filename}_{i}_initial_pptbf_params.txt'))
                    generator.generate(width=400, height=400, filename=pth.join(sub_output_path,f'{filename}_{i}_initial_pptbf_params.txt'))

                # scale resolution parameter
                initial_parameter[3] = int(initial_parameter[3] * 512 / 400)
                save_parameter_file(initial_parameter, pth.join(output_path, filename + '_initial_pptbf_params.txt'))
                generator.generate(width=512, height=512, filename=pth.join(output_path, filename + '_initial_pptbf_params.txt'))

            print('Initial parameter:', initial_parameter)

            optimizer = Optimizer(initial_parameter, target, weight_fft=self.cfg.optim_fft, weight_lbp=self.cfg.optim_lbp,
                                  output_path=output_path, filename=filename, device=self.device, vgg=self.vgg19,
                                  generator=generator, enable_cropping=self.cfg.enable_cropping, mask=mask)

            optimizers.append(optimizer)

        if self.cfg.optimize_closest_query_only:
            closest_optimizer = None
            index = -1
            min_query_loss = np.inf
            for i, optimizer in enumerate(optimizers):
                optimizer.flag = None
                loss = optimizer.loss_func(None)
                if loss < min_query_loss:
                    min_query_loss = loss
                    closest_optimizer = optimizer
                    index = i

            assert closest_optimizer is not None and index != -1
            loss = closest_optimizer.optimize(n_steps=self.cfg.n_step,
                                              max_iter_per_continuous_step=self.cfg.max_iter_per_continuous_step,
                                              max_iter_per_discrete_step=self.cfg.max_iter_per_discrete_step,
                                              ratio=self.cfg.ratio, timer=Timer())
            losses = [np.inf] * len(optimizers)
            losses[index] = loss
        else:
            losses = []
            for optimizer in optimizers:
                loss = optimizer.optimize(n_steps=self.cfg.n_step,
                                          max_iter_per_continuous_step=self.cfg.max_iter_per_continuous_step,
                                          max_iter_per_discrete_step=self.cfg.max_iter_per_discrete_step,
                                          ratio=self.cfg.ratio, timer=Timer())
                losses.append(loss)

        return losses
