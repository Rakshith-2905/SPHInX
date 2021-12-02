import argparse
import math
import os
import glob
import random
import json
import numpy as np
import cv2
import imageio

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils import data
from PIL import Image
from tqdm import tqdm
from metric.ssim import SSIM, PSNR
from utils import *
import copy

import lpips
from model import *

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


class StyleGAN_Inversion():
    def __init__(self, args):

        # Loading the generator
        generator = Generator(
        args.size, args.latent, args.n_mlp)
        args.n_latent = generator.n_latent
        
        generator.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        generator.eval()
        self.generator = generator.to(device)

        self.percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )
        self.ssim_loss = SSIM(window_size = 11)
    
    def init_learners(self, weights_path=None):
        # Initialize the latent projection head and create an optimizer

        Ps = StyleProjection(n_latent = args.n_latent, code_dim = args.latent).to(device)
        Pc = ContentProjection(code_dim=8, style_dim=args.latent).to(device)    
        
        noises_single = self.generator.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(1, 1, 1, 1).normal_())

        if not weights_path:
            for noise in noises:
                noise.requires_grad = True

            optimizer = optim.Adam(Ps.parameters(),)
            optimizer.add_param_group({'params': Pc.parameters()})   
            optimizer.add_param_group({'params': noises})

        if weights_path:
            # Load the learned projection
            results = torch.load(weights_path)
        
            Ps.load_state_dict(results["Ps"])
            Pc.load_state_dict(results["Pc"])
            noises = results['noises']

            true_img = results['true_img']

            return Ps, Pc, noises, true_img
            
        return Ps, Pc, noises, optimizer

    def init_state(self, Ps, Pc):

        latent_z = torch.randn(1, args.n_latent, args.latent, device=device)    

        sample_s = torch.randn(1,8,4,4, device=device)
        semantic_input = Pc(sample_s)

        return Ps(latent_z), semantic_input

    def image_operations(self, img, img_gen):

        if args.ops == 'denoising':
            self.ops = 'denoising/' + f'{args.ops_fac:.2f}' + '/'
            pros_img = add_noise(img.clone(), std=args.ops_fac)
            pros_gen = add_noise(img_gen.clone(), std=args.ops_fac)
        
        elif args.attribute == 'rotate':
            self.ops = 'rotations/'
            rot_imgs = [rot_image(img, angle) for angle in args.attribute_fac]
            return rot_imgs

        else:
            self.ops = None
            pros_img = img
            pros_gen = img_gen

        return pros_img, pros_gen

    def save_results(self, img, img_gen, pros_gt, pros_gen):
        
        gt_ar = make_image(img)
        gen_ar = make_image(img_gen)
        
        save_path = args.output_dir
        if args.ops:
            save_path = args.output_dir + self.ops
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.ops:

            gt_pros_ar = make_image(pros_gt)[0]
            gen_pros_ar = make_image(pros_gen)[0]

            # Save the image        
            img_name = save_path + "pros_gt.png"
            pil_img = Image.fromarray(gt_pros_ar)
            pil_img.save(img_name)

            img_name = save_path + "pros_gen.png"
            pil_img_n = Image.fromarray(gen_pros_ar)
            pil_img_n.save(img_name)

        img_name = save_path + "gen.png"
        pil_img = Image.fromarray(gen_ar[0])
        pil_img.save(img_name)

    def learn_inversion(self, img, Ps, Pc, noises, optimizer):
        
        pros_true, _ = self.image_operations(img, img)
        pbar = tqdm(range(args.step))
        # Iterate for the desired number of steps
        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            # lr = args.lr
            optimizer.param_groups[0]["lr"] = lr

            latent_in, semantic_input = self.init_state(Ps, Pc)

            img_gen, _ = self.generator(latent_in, semantic_input=semantic_input, noise=noises)

            _, pros_gen = self.image_operations(img_gen, img_gen)

            mse_loss = F.mse_loss(pros_gen, pros_true)
            
            if pros_true.shape[-1] < 256:
                p_loss = torch.zeros(1).to(pros_true.device)
            else:
                p_loss = self.percept(downsampler_image_256(pros_gen), 
                                        downsampler_image_256(pros_true))

            loss = args.lpips_weight*p_loss + args.mse * mse_loss
            
            # Optimize the learners
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (   f"p_loss: {p_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        self.save_results(img, img_gen, pros_true, pros_gen)
    
    def learn_att_inversion(self, img):

        att_imgs = self.image_operations(img, None)

        K=len(att_imgs)

        attribute = Attribute_learner(k=K, n_latent=args.n_latent).to(device)

        Ps, Pc, noises, optimizer = self.init_learners()
        optimizer.add_param_group({'params': attribute.parameters()})

        pbar = tqdm(range(args.step))
        # Iterate for the desired number of steps
        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            # lr = args.lr
            optimizer.param_groups[0]["lr"] = lr

            latent_in, semantic_input = self.init_state(Ps, Pc)
            att_latent_ins = attribute(latent_in)

            att_p_loss, att_mse_loss, att_img_gens = [], [], []
            for k, att_img in enumerate(att_imgs):
                img_gen, _ = self.generator(att_latent_ins[k], semantic_input=semantic_input, noise=noises)

                att_img_gens.append(img_gen)
                if img_gen.shape[-1] < 256:
                        att_p_loss.append(torch.zeros(1).to(img_gen.device))
                else:
                    att_p_loss.append(self.percept(downsampler_image_256(img_gen), 
                                            downsampler_image_256(att_img)))

                att_mse_loss.append(F.mse_loss(img_gen, att_img))
            
            alphas = attribute.alpha.clone().cpu()
            mono_reg = 0.
            for k in range(len(alphas)-1):
                mono_reg += torch.max(torch.tensor(0), (torch.tensor(2) - alphas[k]))
                mono_reg += torch.max(torch.tensor(0), (torch.tensor(2)- alphas[k+1]+alphas[k]))

            if len(alphas) == 1:
                mono_reg += torch.max(torch.tensor(0), (torch.tensor(2) - alphas[0]))

            p_loss = torch.mean(torch.stack(att_p_loss))
            mse_loss = torch.mean(torch.stack(att_mse_loss))

            loss = args.lpips_weight*p_loss + args.mse * mse_loss + args.mono_weight * mono_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (   f"pips: {p_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f};"
                    f" alphas: {attribute.alpha.detach().cpu().numpy()}"
                )
            )

        latent_in, semantic_input = self.init_state(Ps, Pc)

        D = attribute.D
        alphas = attribute.alpha.detach().cpu().numpy()
        alphas = np.insert(alphas, 0, 0)
        
        
        if not os.path.exists(args.output_dir + 'attribute_walk/'):
            os.makedirs(args.output_dir + 'attribute_walk/')
                        
        alphas = interpolate_vector(alphas.tolist(), 2)
        gif_imgs = []
        for alpha in alphas:
            att_latent_in = latent_in + (alpha * D/torch.linalg.norm(D, dim=1).unsqueeze(1))
                                
            img_gen, _ = self.generator(att_latent_in, semantic_input=semantic_input, noise=noises)

            rot_img_gen_ar = make_image(img_gen)

            # Save the image
            img_name = args.output_dir + 'attribute_walk/' + f'{alpha:.2f}' + "_alpha.png"
            pil_img = Image.fromarray(rot_img_gen_ar[0])
            pil_img.save(img_name)

            gif_imgs.append(imageio.imread(args.output_dir + 'attribute_walk/' + f'{alpha:.2f}' + "_alpha.png"))

        imageio.mimsave(args.output_dir + 'attribute_walk/' + 'walk.gif', gif_imgs, duration=0.25) 

    def invert(self, img):
 
        img = img.to(device)

        if args.attribute:
            self.learn_att_inversion(img)

        else:
            Ps, Pc, noises, optim = self.init_learners()
            self.learn_inversion(img, Ps, Pc, noises, optim)

            save_dir = args.output_dir
            if self.ops:
                save_dir = args.output_dir + self.ops

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, default='model/stylegan2-ffhq-config-f.pt', help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--step", type=int, default=5000, help="optimize iterations")
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./samples/gen_image/',
        help="path to store the generated image",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default='./samples/true_image/CXR.jpg',
        help="path to ground truth image",
    )
    parser.add_argument("--mono_weight", type=float, default=0.5, help="weight of the monotonic regularizer")
    parser.add_argument("--mse", type=float, default=1.0, help="weight of the mse loss")
    parser.add_argument("--lpips_weight", type=float, default=1.5, help="weight of the lpips loss")    
    parser.add_argument(
        "--ops",
        choices=['denoising'],
        help="select the desired image operation",
    )    
    parser.add_argument("--ops_fac", type=float, default=1.0, help="control factor for the operation")
    parser.add_argument(
        "--attribute",
        choices=['rotate'],
        help="select the dataset",
    )
    parser.add_argument(
        "--attribute_fac",
        nargs="*",
        type=float,
        default=[0, 2], 
        help="The factor of attribute changes to learn"
    )
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8

    print("\n\n************************************************************************\n")
    print(args)

    # ******************************************************************************** #

    image_transforms = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img = Image.open(args.img_path)
    img = image_transforms(img).unsqueeze(0)

    gan = StyleGAN_Inversion(args)

    gan.invert(img)
