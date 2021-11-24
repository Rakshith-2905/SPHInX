import time
import functools
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader

from .fid_score import calculate_frechet_distance
# from .kid_score import polynomial_mmd_averages
from .swd_score import calculate_swd


def get_fake_images_and_acts_I2I(args, Encoder, Align, inception, g_running, code_size, step, alpha, sample_num=5000, batch_size=16, source_loader=None):
	data_loader_y = iter(source_loader)
	real_image_y = next(data_loader_y)
	dataset = TensorDataset(torch.randn(sample_num, code_size))
	loader = DataLoader(dataset, batch_size=real_image_y.shape[0])

	pbar = tqdm(total=sample_num, position=1, leave=False)
	pbar.set_description('Get fake images and acts')

        

        ####
	images = []
	real_images = []
	acts = []

	for gen_in in loader:
		gen_in = gen_in[0].cuda()  # list -> tensor
		with torch.no_grad():
                        try:
                            real_image_y = next(data_loader_y)
                        except (OSError, StopIteration):
                            data_loader_y = iter(source_loader)
                            real_image_y = next(data_loader_y)

                        # E1
                        _, L_feat = Encoder(real_image_y, step=step, alpha=alpha, E1_output_feat=True, RESOLUTION=args.RESOLUTION)
                        # A1
                        L_feat = Align(L_feat)

                        fake_image = g_running(
                                gen_in, step=step, alpha=alpha, E1_output_feat=True, L_feat=L_feat, RESOLUTION=args.RESOLUTION, E1_fea_w=args.E1_fea_w, G_fea_w=args.G_fea_w
                            )
                        out = inception(fake_image)
                        out = out[0].squeeze(-1).squeeze(-1)

		images.append(fake_image.cpu())  # cuda tensor
		real_images.append(real_image_y.cpu())  # cuda tensor

		acts.append(out.cpu().numpy())  # numpy

		pbar.update(len(gen_in))

	images = torch.cat(images, dim=0)  # N x C x H x W
	real_images = torch.cat(real_images, dim=0)  # N x C x H x W
	acts = np.concatenate(acts, axis=0)  # N x d

	return real_images,images, acts

def get_fake_images_and_acts_only_for_fake(inception, g_running, discriminator, align, code_size, step, alpha, sample_num=5000, batch_size=16, encoder=None, E1_output_feat=False, L_feat=None, RESOLUTION=None, E1_fea_w=None):
	dataset = TensorDataset(torch.randn(sample_num, code_size))
	loader = DataLoader(dataset, batch_size=batch_size * torch.cuda.device_count())

	pbar = tqdm(total=sample_num, position=1, leave=False)
	pbar.set_description('Get fake images and acts')

        ####
	images = []
	acts = []
	acts_recon = []
	for gen_in in loader:
		gen_in = gen_in[0].cuda()  # list -> tensor
		with torch.no_grad():
                    
			noise_style=gen_in
			fake_image = g_running(noise_style, step=step, alpha=alpha, E1_output_feat=False)
			noise=g_running.noise
                                            
			fake_predict, D_feature = discriminator(fake_image, step=step, alpha=alpha, E1_output_feat=True)
			D_feature = align(D_feature)

			E1_fea_w={4:1., 8:0., 16:0., 32:0., 64:0.}
			G_fea_w={4:0., 8:1., 16:1., 32:1., 64:1.}
			images_reco_4 = g_running(noise_style, noise=noise, step=step, alpha=alpha, E1_output_feat=True, L_feat=D_feature, RESOLUTION=[64,32,16,8,4], E1_fea_w=E1_fea_w, G_fea_w=G_fea_w)
			out = inception(fake_image)
			out = out[0].squeeze(-1).squeeze(-1)
			out_recon = inception(images_reco_4)
			out_recon = out_recon[0].squeeze(-1).squeeze(-1)

		images.append(fake_image.cpu())  # cuda tensor
		acts.append(out.cpu().numpy())  # numpy
		acts_recon.append(out_recon.cpu().numpy())  # numpy

		pbar.update(len(gen_in))

	images = torch.cat(images, dim=0)  # N x C x H x W
	acts = np.concatenate(acts, axis=0)  # N x d
	acts_recon = np.concatenate(acts_recon, axis=0)  # N x d

	return acts_recon, acts

def get_fake_images_and_acts(inception, g_running, miner, miner_semantic, n_latent, code_size, sample_num=5000, batch_size=16, device=None):
	dataset = TensorDataset(torch.randn(sample_num, n_latent, code_size, device=device))
	#dataset = TensorDataset(torch.randn(sample_num, code_size))
	loader = DataLoader(dataset, batch_size=batch_size * torch.cuda.device_count())

	pbar = tqdm(total=sample_num, position=1, leave=False)
	pbar.set_description('Get fake images and acts')

        ####
	images = []
	acts = []
	for gen_in in loader:
		gen_in = gen_in[0].cuda()  # list -> tensor
		with torch.no_grad():
			if miner is not None:
				if miner_semantic is not None:
					fake_image = g_running(miner(g_running.style(gen_in)), miner_semantic=miner_semantic)[0]# output is image and latent feature
				else:
					fake_image = g_running(miner(g_running.style(gen_in)))[0]# output is image and latent feature
			else:
				fake_image = g_running([g_running.style(gen_in)])[0]# output is image and latent feature
			out = inception(fake_image)
			out = out[0].squeeze(-1).squeeze(-1)

		images.append(fake_image.cpu())  # cuda tensor
		acts.append(out.cpu().numpy())  # numpy

		pbar.update(len(gen_in))

	images = torch.cat(images, dim=0)  # N x C x H x W
	acts = np.concatenate(acts, axis=0)  # N x d

	return images, acts


def compute_time(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		begin = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(f'function: {func}\tcomputation time: {round(end - begin)}s')
		return result
	return wrapper


# @compute_time
def compute_fid(real_acts, fake_acts):
	mu1, sigma1 = (np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False))
	mu2, sigma2 = (np.mean(fake_acts, axis=0), np.cov(fake_acts, rowvar=False))

	fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

	return fid


# @compute_time
def compute_kid(real_acts, fake_acts):
	# size = min(len(real_acts), len(fake_acts))
	# mmds, mmd_vars = polynomial_mmd_averages(real_acts[:size], fake_acts[:size])
	mmds, mmd_vars = polynomial_mmd_averages(real_acts, fake_acts, replace=True)
	kid = mmds.mean()

	# print("mean MMD^2 estimate:", mmds.mean())
	# print("std MMD^2 estimate:", mmds.std())
	# print("MMD^2 estimates:", mmds, sep='\n')
	#
	# print("mean Var[MMD^2] estimate:", mmd_vars.mean())
	# print("std Var[MMD^2] estimate:", mmd_vars.std())
	# print("Var[MMD^2] estimates:", mmd_vars, sep='\n')

	return kid


# @compute_time
def compute_swd(real_images, fake_images):
	# size = min(len(real_images), len(fake_images))
	# swd = calculate_swd(real_images[:size], fake_images[:size], device="cuda")
	swd = calculate_swd(real_images, fake_images, device="cuda", enforce_balance=False)

	return swd


