from comet_ml import OfflineExperiment, Experiment
import torchvision
import numpy as np
import math
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from torch import nn
import hydra
import os
import logging
import random
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import MVU_Estimator_Brain, MVU_Estimator_Knees, MVU_Estimator_Stanford_Knees, MVU_Estimator_Abdomen
import multiprocessing
import PIL.Image
from torch.utils.data.distributed import DistributedSampler
from utils import *
import matplotlib.pyplot as plt

from ncsnv2.models import get_sigmas
from ncsnv2.models.ema import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
import argparse

from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2 

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter()

def normalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling

def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling


class LangevinOptimizer(torch.nn.Module):
    def __init__(self, config, logger, project_dir='./', experiment=None):
        super().__init__()

        self.config = config

        self.langevin_config = self._dict2namespace(self.config['langevin_config'])
        self.device = config['device']
        self.langevin_config.device = config['device']

        self.project_dir = project_dir
        self.score = NCSNv2Deepest(self.langevin_config).to(self.device)
        self.sigmas_torch = get_sigmas(self.langevin_config)

        self.sigmas = self.sigmas_torch.cpu().numpy()

        states = torch.load(os.path.join(project_dir, config['gen_ckpt']))#, map_location=self.device)

        self.score = torch.nn.DataParallel(self.score)

        self.score.load_state_dict(states[0], strict=True)
        if self.langevin_config.model.ema:
            ema_helper = EMAHelper(mu=self.langevin_config.model.ema_rate)
            ema_helper.register(self.score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.score)
        del states

        self.index = 0
        self.experiment = experiment
        self.logger = logger
        self.nrmse = []
        self.ssim = []
        self.noise = []
        self.p = []
        self.m = []
        self.step = []
        self.samples = []

    def _dict2namespace(self,langevin_config):
        namespace = argparse.Namespace()
        for key, value in langevin_config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    def _initialize(self):
        self.gen_outs = []

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x


    def _sample(self, y):
        samples =[0.00037671177415177226, 2426.02783203125, 2660.712890625, 1724.640380859375, 942.825927734375, 643.960205078125, 416.3608703613281, 232.9564666748047, 162.61135864257812, 92.96804809570312, 63.5211067199707, 38.948081970214844, 25.89923667907715, 14.312627792358398, 9.692362785339355, 6.0063958168029785, 4.163099765777588, 2.5627479553222656, 2.0072121620178223, 1.5618263483047485, 1.2954530715942383, 1.141144037246704, 1.1409183740615845, 1.0744963884353638, 1.0691808462142944]
        p =[0.017614595592021942, 0.02940373122692108, 0.04492798075079918, 0.07432198524475098, 0.11127711832523346, 0.16438837349414825, 0.29640406370162964, 0.4270467758178711, 0.6764713525772095, 1.1062217950820923, 1.5645519495010376, 2.4600677490234375, 3.9031927585601807, 6.137128829956055, 10.717514038085938, 18.466217041015625, 25.44629669189453, 35.67254638671875, 60.55537033081055, 96.55986785888672, 138.7302703857422, 220.4004669189453, 342.4575500488281, 510.0615234375]
        m= [0.1033674031496048, 0.17199289798736572, 0.22938458621501923, 0.3764371871948242, 0.5409281253814697, 0.8915922045707703, 1.268998622894287, 2.1148107051849365, 3.4807238578796387, 5.541201591491699, 7.950469970703125, 11.426004409790039, 20.549121856689453, 31.36182403564453, 45.98844528198242, 76.54082489013672, 111.30801391601562, 188.6122283935547, 259.5984191894531, 426.67315673828125, 697.49658203125, 972.858154296875, 1501.4349365234375, 2262.189697265625]
        noise =[1560.111572265625, 992.1370849609375, 663.839111328125, 418.6387634277344, 271.6714782714844, 160.48619079589844, 103.32732391357422, 70.52400970458984, 42.81882095336914, 27.635183334350586, 17.686912536621094, 11.595663070678711, 6.751256465911865, 4.131176471710205, 3.0070407390594482, 1.667025089263916, 1.09114408493042, 0.7393677234649658, 0.4590584933757782, 0.28390973806381226, 0.19987154006958008, 0.1299794465303421, 0.07399361580610275, 0.04815861955285072]
        step =[61781.45253673096, 24961.455601867867, 10085.134227924347, 4074.6793774303915, 1646.2857140091062, 665.1459374108344, 268.7378115680814, 108.57766858081818, 43.86844756579418, 17.724092813920976, 7.161034643199342, 2.893261079863878, 1.1689591359393672, 0.47229260422802766, 0.19081955916476437, 0.07709650545695622, 0.03114916841975519, 0.012585143890616747, 0.005084754423049344, 0.0020543847291683505, 0.0008300295584240416, 0.0003353553940233269, 0.0001354930606719364, .0005474302756593446]
        x=np.arange(24)
        fig, axs = plt.subplots(5)
        im=axs[0].plot(np.arange(25), samples)
        axs[0].set_title('Samples')

        im=axs[1].plot(x, p)
        axs[1].set_title('P Grad')
        
        im=axs[2].plot(x, m)
        axs[2].set_title('M Grad')
        
        im=axs[3].plot(x, noise)
        axs[3].set_title('Noise')
        
        im=axs[4].plot(x, step)
        axs[4].set_title('Step Size')
        fig.tight_layout()
        fig.savefig('Scaling.jpg')
        print('check')
        
        ref, mvue, maps, batch_mri_mask = y
        estimated_mvue = torch.tensor(
            get_mvue(ref.cpu().numpy(),
            maps.cpu().numpy()), device=ref.device)
        self.logger.info(f"Running {self.langevin_config.model.num_classes} steps of Langevin.")
        pbar = tqdm(range(self.langevin_config.model.num_classes), disable=(self.config['device'] != 0))
        pbar_labels = ['class', 'step_size', 'error', 'mean', 'max']
        step_lr = self.langevin_config.sampling.step_lr
        forward_operator = lambda x: MulticoilForwardMRI(self.config['orientation'])(torch.complex(x[:, 0], x[:, 1]), maps, batch_mri_mask)


        samplesRanging = torch.rand(y[0].shape[0], self.langevin_config.data.channels,
                                 self.config['image_size'][0],
                                 self.config['image_size'][1], device=self.device)
        samples=torch.view_as_real(estimated_mvue).permute(0, 3, 1,2).type(torch.cuda.FloatTensor)

        samples= normalize(samples, samplesRanging)
        self.samples.append(torch.max(samples).item())
        print(self.samples)
        with torch.no_grad():
            for c in pbar:
#                 if c <= self.config['start_iter']:
#                     continue
                if c <= 1800:
                    n_steps_each = 3
                else:
                    n_steps_each = self.langevin_config.sampling.n_steps_each
                sigma = self.sigmas[c]
                labels = torch.ones(samples.shape[0], device=samples.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

                for s in range(n_steps_each):
                    noise = torch.randn_like(samples) * np.sqrt(step_size * 2)
                    # get score from model
                    p_grad = self.score(samples, labels)

                    # get measurements for current estimate
                    meas = forward_operator(normalize(samples, estimated_mvue))
                    # compute gradient, i.e., gradient = A_adjoint * ( y - Ax_hat )
                    # here A_adjoint also involves the sensitivity maps, hence the pointwise multiplication
                    # also convert to real value since the ``complex'' image is a real-valued two channel image
                    meas_grad = torch.view_as_real(torch.sum(self._ifft(meas-ref) * torch.conj(maps), axis=1) ).permute(0,3,1,2)
                    # re-normalize, since measuremenets are from a normalized estimate
                    meas_grad = unnormalize(meas_grad, estimated_mvue)
                    # convert to float incase it somehow became double
                    meas_grad = meas_grad.type(torch.cuda.FloatTensor)
                    meas_grad /= torch.norm( meas_grad )
                    meas_grad *= torch.norm( p_grad )
                    meas_grad *= self.config['mse']
                    # combine measurement gradient, prior gradient and noise
                    samples = samples + step_size * (p_grad - meas_grad) + noise

                    # compute metrics
                    metrics = [c, step_size, (meas-ref).norm(), (p_grad-meas_grad).abs().mean(), (p_grad-meas_grad).abs().max()]
                    update_pbar_desc(pbar, metrics, pbar_labels)
                    # if nan, break
                    if np.isnan((meas - ref).norm().cpu().numpy()):
                        return normalize(samples, estimated_mvue)
                if c%100 == 0:
                    print('happening')
                    self.samples.append(torch.max(samples).item())
                    self.p.append(torch.max(p_grad).item())
                    self.m.append(torch.max(meas_grad).item())
                    self.step.append(step_size.item())
                    self.noise.append(torch.max(noise).item())
                if not self.config['save_images']:
#                     if (c+1) % self.config['save_iter'] ==0 :
                    if (c) % 100 ==0 :
                        print('samples', torch.max(samples).item())
                        print('p_grad', torch.max(p_grad).item())
                        print('meas_grad', torch.max(meas_grad).item())
                        print('step size', step_size)
                        print('noise', torch.max(noise).item())
                        img_gen = samples #normalize(samples, estimated_mvue)
                        to_display = torch.view_as_complex(img_gen.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs()   
                        #NEW CODE
                        to_displayP = torch.view_as_complex(p_grad.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs().type(torch.cuda.FloatTensor)   
                        to_displayM = torch.view_as_complex(meas_grad.permute(0, 2, 3, 1).reshape(-1, self.config['image_size'][0], self.config['image_size'][1], 2).contiguous()).abs().type(torch.cuda.FloatTensor)
                        ##
                        self.samples.append(torch.max(samples).item())
                        self.p.append(torch.max(p_grad).item())
                        self.m.append(torch.max(meas_grad).item())
                        self.step.append(step_size.item())
                        self.noise.append(torch.max(noise).item())
                        if self.config['anatomy'] == 'brain':
                            # flip vertically
                            to_display = to_display.flip(-2)
                        elif self.config['anatomy'] == 'knees':
                            # flip vertically and horizontally
                            to_display = to_display.flip(-2)
                            to_display = to_display.flip(-1)
                        elif self.config['anatomy'] == 'stanford_knees':
                            # do nothing
                            pass
                        elif self.config['anatomy'] == 'abdomen':
                            # flip horizontally
                            to_display = to_display.flip(-1)
                        else:
                            pass
                        for i, exp_name in enumerate(self.config['exp_names']):
                            if self.config['repeat'] == 1:
                                file_name = f'{exp_name}_R={self.config["R"]}_{c}.jpg'
                                title=f'{exp_name}_R={self.config["R"]}_{c}.jpg'
                                imageReg=to_display[i:i+1][0].cpu().numpy()
                                imageP=to_displayP[i:i+1][0].cpu().numpy()
                                imageM=to_displayM[i:i+1][0].cpu().numpy()
                                ##reconstruction error
                                mvueConstructed = mvue.flip(-2)[i:i+1][0][0].cpu().numpy()
                                (score, diff) = structural_similarity(imageReg, mvueConstructed, full=True)
                                self.ssim.append(score)
                                self.nrmse.append((np.sqrt(np.mean(np.square(mvueConstructed-imageReg)))/(mvueConstructed.max() - mvueConstructed.min())).real)
                                print('NRMSE', self.nrmse)
                                print('SSIM', self.ssim)
                                # 3 x 2 gif of reconstruction by fourier
                                fig, axs = plt.subplots(3,2)
                                im=axs[0,0].imshow(imageReg)
                                axs[0,0].set_title(f'Full Reconstruction_{c}')
                                fig.colorbar(im, ax=axs[0,0])
                                im=axs[1,0].imshow(imageP)
                                axs[1,0].set_title(f'PGRAD_{c}')
                                fig.colorbar(im, ax=axs[1,0])
                                im=axs[2,0].imshow(imageM)
                                axs[2,0].set_title(f'MGRAD_{c}')
                                fig.colorbar(im, ax=axs[2,0])
                                
                                
                                to_display = self._fft(to_display).real  
                                imageReg=to_display[i:i+1][0].cpu().numpy()
                                to_displayP = self._fft(to_displayP).real  
                                imageP=to_displayP[i:i+1][0].cpu().numpy()
                                to_displayM = self._fft(to_displayM).real  
                                imageM=to_displayM[i:i+1][0].cpu().numpy()
                                im=axs[0,1].imshow(np.log(np.abs(imageReg)+1e-8),cmap='gray')
                                axs[0,1].set_title(f'Fourier Full_{c}')
                                fig.colorbar(im, ax=axs[0,1])
                                im=axs[1,1].imshow(np.log(np.abs(imageP)+1e-8),cmap='gray')
                                axs[1,1].set_title(f'Fourier PGRAD_{c}')
                                fig.colorbar(im, ax=axs[1,1])
                                im=axs[2,1].imshow(np.log(np.abs(imageM)+1e-8),cmap='gray')
                                axs[2,1].set_title(f'Fourier MGRAD_{c}')
                                fig.colorbar(im, ax=axs[2,1])
                                fig.tight_layout()
                                fig.savefig(title)
## GIF CODE ENDS HERE, start difference plots
#                                 fig, axs = plt.subplots(2,2)
#                                 im=axs[0,0].imshow(np.abs(imageReg-mvueConstructed))
#                                 axs[0,0].set_title(f'Reconstruction Error_{c}')
#                                 fig.colorbar(im, ax=axs[0,0])
                                
#                                 im=axs[0,1].imshow(np.abs(mvueConstructed))
#                                 axs[0,1].set_title(f'Reference_{c}')
#                                 fig.colorbar(im, ax=axs[0,1])
                                
#                                 im=axs[1,1].imshow(np.abs(imageReg))
#                                 axs[1,1].set_title(f'Reconstruction Image_{c}')
#                                 fig.colorbar(im, ax=axs[1,1])
                                
#                                 im=axs[1,0].imshow(np.abs(imageP-imageM))
#                                 axs[1,0].set_title(f'PGrad-MGrad_{c}')
#                                 fig.colorbar(im, ax=axs[1,0])
#                                 fig.tight_layout()
#                                 fig.savefig(f'{exp_name}_R={self.config["R"]}_{c}.jpg')
#                                 save_images(to_display[i:i+1], file_name, normalize=True)
                                #
                                if self.experiment is not None:
                                    self.experiment.log_image(file_name)
                            else:
                                for j in range(self.config['repeat']):
                                    file_name = f'{exp_name}_R={self.config["R"]}_sample={j}_{c}.jpg'
                                    save_images(to_display[j:j+1], file_name, normalize=True)
                                    if self.experiment is not None:
                                        self.experiment.log_image(file_name)

                        # uncomment below if you want to save intermediate samples, they are logged to CometML in the interest of saving space
                        # intermediate_out = samples
                        # intermediate_out.requires_grad = False
                        # self.gen_outs.append(intermediate_out)
                # if c>=0:
                #     break
#         x=np.arange(24)
#         fig, axs = plt.subplots(2)
#         im=axs[0].plot(x,nrmse)
#         axs[0].set_title("RMSE")

#         im=axs[1].plot(x, ssim)
#         axs[1].set_title("SSIM")
#         fig.tight_layout()
#         fig.savefig("Similarity.jpg")
        print('samples', self.samples)
        print('p', self.p)
        print('m', self.m)
        print('noise', self.noise)
        print('setp', self.step)
     
        x=np.arange(24)
        fig, axs = plt.subplots(5)
        im=axs[0].plot(np.arange(25), self.samples)
        axs[0].set_title("Samples")

        im=axs[1].plot(x, self.p)
        axs[1].set_title("P Grad")
        
        im=axs[2].plot(x, self.m)
        axs[2].set_title("M Grad")
        
        im=axs[3].plot(x, self.noise)
        axs[3].set_title("Noise")
        
        im=axs[4].plot(x, self.step)
        axs[4].set_title("Step Size")
        fig.tight_layout()
        fig.savefig("Scaling.jpg")
        return normalize(samples, estimated_mvue)



    def sample(self, y):
        self._initialize()
        mvue = self._sample(y)

        outputs = []
        for i in range(y[0].shape[0]):
            outputs_ = {
                'mvue': mvue[i:i+1],
                # uncomment below if you want to return intermediate output
                # 'gen_outs': self.gen_outs
            }
            outputs.append(outputs_)
        return outputs

def mp_run(rank, config, project_dir, working_dir, files):
    if config['multiprocessing']:
        mp_setup(rank, config['world_size'])
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    logger = MpLogger(logger, rank)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    logger.info(f'Logging to {working_dir}')
    if rank == 0 and not config['debug']:
        # uncomment the following to log the experiment offline
        # will need to add api key to see experiments online
        #api_key = None
        #project_name = config['anatomy']
        #experiment = Experiment(api_key,
        #                        project_name=project_name,
        #                        auto_output_logging='simple')
        project_name = config['anatomy']
        experiment = OfflineExperiment(
                                project_name=project_name,
                                auto_output_logging='simple',
                                offline_directory="./outputs")

        experiment.log_parameters(config)
        pretty(config)
    else:
        experiment = None

    config['device'] = rank
    # load appropriate dataloader
    if config['anatomy'] == 'knees':
        dataset = MVU_Estimator_Knees(files,
                            input_dir=config['input_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'])
    elif config['anatomy'] == 'stanford_knees':
        dataset = MVU_Estimator_Stanford_Knees(files,
                            input_dir=config['input_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'])
    elif config['anatomy'] == 'abdomen':
        dataset = MVU_Estimator_Abdomen(
                            input_dir=config['input_dir'],
                            maps_dir=config['maps_dir'],
                            project_dir=project_dir,
                            image_size = config['image_size'],
                            R=config['R'],
                            pattern=config['pattern'],
                            orientation=config['orientation'],
                            rotate=config['rotate'])

    elif config['anatomy'] == 'brain':
        dataset = MVU_Estimator_Brain(files,
                                input_dir=config['input_dir'],
                                maps_dir=config['maps_dir'],
                                project_dir=project_dir,
                                image_size = config['image_size'],
                                R=config['R'],
                                pattern=config['pattern'],
                                orientation=config['orientation'])
    else:
        raise NotImplementedError('anatomy not implemented, please write dataloader to process kspace appropriately')

    sampler = DistributedSampler(dataset, rank=rank, shuffle=True) if config['multiprocessing'] else None
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])


    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config['batch_size'],
                                         sampler=sampler,
                                         shuffle=True if sampler is None else False)

    langevin_optimizer = LangevinOptimizer(config, logger, project_dir, experiment=experiment)
    if config['multiprocessing']:
        langevin_optimizer = DDP(langevin_optimizer, device_ids=[rank]).module
    langevin_optimizer.to(rank)

    for index, sample in enumerate(tqdm(loader)):
        '''
                    ref: one complex image per coil
                    mvue: one complex image reconstructed using the coil images and the sensitivity maps
                    maps: sensitivity maps for each one of the coils
                    mask: binary valued kspace mask
        '''

        ref, mvue, maps, mask = sample['ground_truth'], sample['mvue'], sample['maps'], sample['mask']
        # uncomment for meniscus tears
        # exp_name = sample['mvue_file'][0].split('/')[-1] + '|langevin|' + f'slide_idx_{sample["slice_idx"][0].item()}'
        # # if exp_name != 'file1000425.h5|langevin|slide_idx_22':
        # if exp_name != 'file1002455.h5|langevin|slide_idx_26':
        #     continue

        # move everything to cuda
        ref = ref.to(rank).type(torch.complex128)
        mvue = mvue.to(rank)
        maps = maps.to(rank)
        mask = mask.to(rank)
        estimated_mvue = torch.tensor(
            get_mvue(ref.cpu().numpy(),
            maps.cpu().numpy()), device=ref.device)


        exp_names = []
        for batch_idx in range(config['batch_size']):

            exp_name = sample['mvue_file'][batch_idx].split('/')[-1] + '|langevin|' + f'slide_idx_{sample["slice_idx"][batch_idx].item()}'
            exp_names.append(exp_name)
            print(exp_name)
            if config['save_images']:
                file_name = f'{exp_name}_R={config["R"]}_estimated_mvue.jpg'
                save_images(estimated_mvue[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
                if experiment is not None:
                    experiment.log_image(file_name)

                file_name = f'{exp_name}_input.jpg'
                save_images(mvue[batch_idx:batch_idx+1].abs().flip(-2), file_name, normalize=True)
                if experiment is not None:
                    experiment.log_image(file_name)

        langevin_optimizer.config['exp_names'] = exp_names
        if config['repeat'] > 1:
            repeat = config['repeat']
            ref, mvue, maps, mask, estimated_mvue = ref.repeat(repeat,1,1,1), mvue.repeat(repeat,1,1,1), maps.repeat(repeat,1,1,1), mask.repeat(repeat,1), estimated_mvue.repeat(repeat,1,1,1)
        outputs = langevin_optimizer.sample((ref, mvue, maps, mask))


        for i, exp_name in enumerate(exp_names):
            if config['repeat'] == 1:
                torch.save(outputs[i], f'{exp_name}_R={config["R"]}_outputs.pt')
            else:
                for j in range(config['repeat']):
                    torch.save(outputs[j], f'{exp_name}_R={config["R"]}_sample={j}_outputs.pt')

        # todo: delete after testing
        if index >= 0:
            break

    if config['multiprocessing']:
        mp_cleanup()

@hydra.main(config_path='configs')
def main(config):
    """ setup """

    working_dir = os.getcwd()
    project_dir = hydra.utils.get_original_cwd()

    folder_path = os.path.join(project_dir, config['input_dir'])
    if config['anatomy'] == 'stanford_knees':
        files = get_all_files(folder_path, pattern=f'*R{config["R"]}*.h5')
    else:
        files = get_all_files(folder_path, pattern='*.h5')

    if not config['multiprocessing']:
        mp_run(0, config, project_dir, working_dir, files)
    else:
        mp.spawn(mp_run,
                args=(config, project_dir, working_dir, files),
                nprocs=config['world_size'],
                join=True)


if __name__ == '__main__':
    main()