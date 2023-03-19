#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim, torch.nn.init
import matplotlib.pyplot as plt
import math, os, tqdm, copy
import numpy as np


# In[2]:


z_dim = 128
z_dim2 = z_dim * 2
z_dim22 = z_dim2 * 2
h_dim = 64
loss_weight = 1
kl_weight = 1
normal_kl_weight = 1e-10
root_normal_kl_weight = 1e-6

bn_momentum = 0.1

reduction_type = 'mean'

torch.set_default_dtype(torch.float32)


# In[3]:


class Conv2d(nn.Module):
	def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True, scale=1, groups=1):
		super().__init__()

		self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=bias, groups=groups)
		fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
		scale /= (fan_in + fan_out) / 2
		bound = math.sqrt(3 * scale)
		nn.init.uniform_(self.conv.weight, -bound, bound)

	def forward(self, x):
		return self.conv(x)


class SENet(nn.Module):
	def __init__(self, channels):
		super().__init__()
		
		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1,1)),
			Conv2d(channels, channels // 4, kernel_size=1, scale=1e-6),
			nn.SiLU(),
			Conv2d(channels // 4, channels, kernel_size=1, scale=1e-6),
			nn.SiLU()
		)
		
	def forward(self, x):
		se = self.se(x)
		return x * se
	

class ResNetGenerative(nn.Module):
	def __init__(self, channels, channels_out=None, kernel_size=3, padding=None, upsample_scale=1, groups=16, sample_mode='nearest'
				, ex=2):
		super().__init__()
		
		if upsample_scale != 1:
			if sample_mode == 'bilinear':
				scale_sample = nn.Upsample(scale_factor=upsample_scale, mode=sample_mode, align_corners=True)
			else:
				scale_sample = nn.Upsample(scale_factor=upsample_scale, mode=sample_mode)
		else:
			scale_sample = Identity()
		
		if channels_out is None:
			channels_out = channels
			self.re_arrange = scale_sample
		else:
			self.re_arrange = nn.Sequential(
				scale_sample,
				Conv2d(channels, channels_out, 1, scale=1e-10),
					)
			
		if padding is None:
			padding = (kernel_size - 1) // 2
			
		self.E = ex
			
		self.double_conv = nn.Sequential(
			#nn.GroupNorm(groups, channels),
			nn.BatchNorm2d(channels, eps=1e-5, momentum=bn_momentum),
			scale_sample,
			Conv2d(channels, channels * self.E, 1, bias=False),
			#nn.GroupNorm(groups, channels * self.E),
			nn.BatchNorm2d(channels * self.E, eps=1e-5, momentum=bn_momentum),
			nn.SiLU(),
			Conv2d(channels * self.E, channels * self.E, kernel_size, padding=padding, bias=False, groups=channels, scale=1e-6),
			#nn.GroupNorm(groups, channels * self.E),
			nn.BatchNorm2d(channels * self.E, eps=1e-5, momentum=bn_momentum),
			nn.SiLU(),
			Conv2d(channels * self.E, channels_out, 1, bias=False, scale=1e-6),
			#nn.GroupNorm(groups, channels_out),
			nn.BatchNorm2d(channels_out, eps=1e-5, momentum=bn_momentum),
			SENet(channels_out)
		)

	def forward(self, x):
		return self.re_arrange(x) + self.double_conv(x)


class ResNetEncoder(nn.Module):
	def __init__(self, channels, channels_out=None, kernel_size=3, stride=1, padding=None, groups=16):
		super().__init__()
		
		if padding is None:
			padding = (kernel_size - 1) // 2
		
		if channels_out is None:
			channels_out = channels
			self.re_arrange = Identity()
		else:
			self.re_arrange = Conv2d(channels, channels_out, kernel_size, stride, padding, scale=1e-6)
		
		self.double_conv = nn.Sequential(
			#nn.GroupNorm(groups, channels),
			nn.BatchNorm2d(channels, eps=1e-5, momentum=bn_momentum),
			nn.SiLU(),
			Conv2d(channels, channels, kernel_size, stride, padding, bias=False),
			#nn.GroupNorm(groups, channels),
			nn.BatchNorm2d(channels, eps=1e-5, momentum=bn_momentum),
			nn.SiLU(),
			Conv2d(channels, channels_out, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels_out, eps=1e-5, momentum=bn_momentum),
			SENet(channels_out)
		)
		
	def forward(self, x):
		return self.double_conv(x) + self.re_arrange(x)


class Identity(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x


class SelfAttention(nn.Module):
	def __init__(self, channel, groups=16) -> None:
		super().__init__()

		#self.norm = nn.GroupNorm(groups, channel)
		self.norm = nn.BatchNorm2d(channel)
		self.qkv = Conv2d(channel, channel * 3, 3, padding=1, bias=False)
		self.out = Conv2d(channel, channel, 3, padding=1, scale=1e-6, bias=False)

	def forward(self, x):
		batch, channels, height, width = x.shape
		x_in = self.norm(x)
		q, k, v = torch.chunk(self.qkv(x_in), 3, 1)
		attn = torch.einsum('bcyx, bchw -> bcyxhw', q, k).contiguous() / math.sqrt(channels)
		attn = attn.view(batch, channels, height, width, -1)
		attn = torch.softmax(attn, dim=-1)
		attn = attn.view(batch, channels, height, width, height, width)
		v = torch.einsum('bchwyx, bcyx -> bchw', attn, v).contiguous()
		v = v.view(batch, channels, height, width)
		return x + self.out(v)


# In[4]:


class DynamicConv(nn.Module):
	def __init__(self, channels, cout=None):
		super().__init__()
		
		if cout == None:
			self.channels_out = channels
			self.outer = Identity()
		else:
			self.channels_out = cout
			self.outer = Conv2d(channels, cout, 1, scale=1e-3, bias=False)
		
		self.conv_embs = 16
		
		print(channels)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.create_attn = nn.Sequential(
			Conv2d(channels, 128, 1, scale=1e-6, bias=False),
			nn.BatchNorm2d(128),
			nn.SiLU(),
			Conv2d(128, self.conv_embs, 1, scale=1e-6, bias=False),
			nn.BatchNorm2d(self.conv_embs),
			nn.Softmax(dim=-1)
		)
		
		self.norm = nn.BatchNorm2d(self.channels_out)
		
		self.conv_para = nn.Parameter(
			torch.randn(self.conv_embs, self.channels_out, channels, 3, 3) * 1e-10
		)
		
	def forward(self, origin):
		batch, channels, height, width = origin.shape
		x = self.pool(origin)
		attn = self.create_attn(x).view(batch, self.conv_embs, 1, 1, 1, 1)
		para = self.conv_para.repeat(batch, 1, 1, 1, 1, 1)
		weight = attn * para
		weight = torch.mean(torch.sum(weight, dim=1), dim=0)
		#kl_norm = -torch.sum(1 + weight - torch.exp(weight))
		#weight = torch.randn_like(weight) * torch.exp(weight / 2)
		x = F.conv2d(origin, weight, padding=1)
		x = self.norm(x)
		return x + self.outer(origin)


class Attacher(nn.Module):
	def __init__(self):
		super().__init__()
		
		#self.convertor = nn.ModuleList([
		#	nn.Linear(4**2, 4**2),
		#	nn.Linear(16**2, 16**2),
		#	nn.Linear(32**2, 32**2),
		#	nn.Linear(128**2, 128**2)
		#])
		self.convertor = nn.ModuleList([
			DynamicConv(128),
			DynamicConv(32),
			DynamicConv(16),
			DynamicConv(8)
		])
		
	def forward(self, latents_enc, latents_dec):
		loss = 0
		for i in range(4):
			#latents_enc[i], latents_dec[i] = latents_enc[i].detach(), latents_dec[i].detach()
			batch, channels, height, width = latents_dec[i].shape
			#print(channels, i)
			l = self.convertor[i](latents_dec[i])
			mu1, log_var1 = torch.chunk(latents_enc[i], 2, dim=1)
			mu2, log_var2 = torch.chunk(l, 2, dim=1)
			d_log_var, d_mu = log_var1 - log_var2, mu1 - mu2
			loss = loss - torch.sum(1 + d_log_var - torch.pow(d_mu, 2) - torch.exp(d_log_var))
		return loss
	
	def process(self, layer_idx, latent):
		return self.convertor[layer_idx](latent)


# In[5]:


class AutoEncoder(nn.Module):
	def __init__(self, convertor=None):
		super().__init__()
		
		print('input: 3*128*128')
		print('mult: 2, 4, 4, 2')
		print('latent_num=4')
		
		self.dec_convertor = nn.ModuleList([
			DynamicConv(128 + 64, 128),
			DynamicConv(32 + 16, 32),
			DynamicConv(16 + 8, 16),
			DynamicConv(8 + 4, 8),
			DynamicConv(128 + h_dim, 128)
		])
		
		self.enc_convertor = nn.ModuleList([
			DynamicConv(128),
			DynamicConv(32),
			DynamicConv(16),
			DynamicConv(8)
		])
		
		self.preprocess = Conv2d(3, 4, 1, scale=1e-6)
		
		self.encoder = nn.ModuleList([
			ResNetEncoder(4, 8, 3, 1, 1, groups=2), # 128*128
			Identity(),#SelfAttention(16, 4),
			ResNetEncoder(8, 16, 8, 4, 2, groups=2), # 32*32
			SelfAttention(16, 4),
			ResNetEncoder(16, 32, 4, 2, 1, groups=4), # 16*16
			SelfAttention(32),
			ResNetEncoder(32, 128, 8, 4, 2), # 4*4
			SelfAttention(128),
			ResNetEncoder(128, 256, 4, 2, 1), # 2*2
			Identity(),
		])
		
		
		self.h = nn.Parameter(torch.randn(h_dim, 2, 2))
		self.decoder = nn.ModuleList([
			ResNetGenerative(128, upsample_scale=2, ex=6),
			SelfAttention(128),
			ResNetGenerative(128, 32, 5, upsample_scale=4, ex=3),
			SelfAttention(32),
			ResNetGenerative(32, 16, 5, upsample_scale=2, groups=4, ex=3),
			SelfAttention(16, 4),
			ResNetGenerative(16, 8, 7, upsample_scale=4, groups=2, ex=6),
			Identity(),#SelfAttention(24, 3)
		])
		self.postprocess = Conv2d(8, 3, 1, scale=1e-6, bias=False)
	
	def sample(self, z2):
		mu, log_var = torch.chunk(z2, 2, dim=1)
		return torch.randn_like(mu) * torch.exp(log_var / 2) + mu
	
	def calc_kl(self, dist_enc, dist_dec):
		mu1, log_var1 = torch.chunk(dist_enc, 2, dim=1)
		mu2, log_var2 = torch.chunk(dist_dec, 2, dim=1)
		d_log_var, d_mu = log_var1 - log_var2, mu1 - mu2
		return -torch.sum(1 + d_log_var - torch.pow(d_mu, 2) - torch.exp(d_log_var))
	
	def calc_kl_normal(self, dist):
		mu, log_var = torch.chunk(dist, 2, dim=1)
		return -torch.sum(1 + log_var - torch.pow(mu, 2) - torch.exp(log_var))
	
	def calc_div(self, latents, dec_latents, mu, log_var):
		log_var = log_var.view(batch_size, z_dim, -1)
		mu = mu.view(batch_size, z_dim, -1)
		kl_div = -torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
		for i in range(latent_nums):
			(mu, log_var) = latents[i]
			(mu1, log_var1) = dec_latents[i]
			mu, log_var = mu.view(batch_size, z_dim, -1), log_var.view(batch_size, z_dim, -1)
			mu1, log_var1 = mu1.view(batch_size, z_dim, -1), log_var1.view(batch_size, z_dim, -1)
			d_log_var, d_mu = log_var - log_var1, mu - mu1
			kl_div = kl_div - torch.mean(
				torch.sum(1 + d_log_var - torch.pow(d_mu, 2) - torch.exp(d_log_var), dim=-1)
			)
		return kl_div
	
	def enc_combiner(self, enc_dist, dec_dist, layer_idx):
		return (enc_dist + self.enc_convertor[layer_idx](dec_dist)) / 2
	
	def dec_combiner(self, z, dec_data, layer_idx):
		l = torch.cat([dec_data, z], dim=1)
		return self.dec_convertor[layer_idx](l)
	
	def forward(self, x):
		x = x * 2 - 1
		l = self.preprocess(x)
		
		#encoder bottom start
		latents = []
		for i in range(5):
			l = self.encoder[i*2](l)
			latents.append(l)
			l = self.encoder[i*2+1](l)
		#encoder top reach
		
		del latents[-1]
		kl_div = self.calc_kl_normal(l) * root_normal_kl_weight
		ztop = self.sample(l) #top z
		
		#encoder-decoder top start
		l = self.dec_combiner(self.h.repeat(x.shape[0], 1, 1, 1), ztop, -1)
		latents.reverse()
		for i in range(4):
			l = self.decoder[i*2](l)
			l = self.decoder[i*2+1](l)
			latents[i] = self.enc_combiner(latents[i], l, i)
			latents[i] = torch.clamp(latents[i], -30, 20)
			kl_div = kl_div + self.calc_kl_normal(latents[i]) * normal_kl_weight
			z = self.sample(latents[i])
			l = self.dec_combiner(l, z, i)
			#print(l.shape)
		#encoder-decoder bottom reach
		
		
		#decoder top start
		l = self.dec_combiner(self.h.repeat(z.shape[0], 1, 1, 1), ztop, -1)
		for i in range(4):
			l = self.decoder[i*2](l)
			l = self.decoder[i*2+1](l)
			l = torch.clamp(l, -30, 20)
			#l_z = self.convertor[i](l)
			kl_div = kl_div + self.calc_kl(latents[i], l) * kl_weight + self.calc_kl_normal(l) * normal_kl_weight
			z = self.sample(l)
			l = self.dec_combiner(l, z, i)
		#decoder bottom reach
		
		l = self.postprocess(l)
		
		rec_loss = F.mse_loss(x, l, reduction='sum')
		l = torch.clamp((l + 1) / 2, 0, 1)
		
		return l, rec_loss, kl_div
	
	def encode(self, x):
		x = x * 2 - 1
		x = self.preprocess(x)
		for m in self.encoder:
			x = m(x)
		return torch.chunk(x, 2, dim=1)
	
	def decode(self, z):
		z = self.dec_combiner(self.h.repeat(z.shape[0], 1, 1, 1), z)
		for i in range(4):
			z = self.decoder[i*2](z)
			z = torch.clamp(z, -30, 20)
			z = self.decoder[i*2+1](z)
			zsample = self.sample(z)
			z = self.dec_combiner(z, zsample)
		z = self.postprocess(z)
		z = torch.clamp((z + 1) / 2, 0, 1)
		return z
	
	def encode_debug(self, x):
		x = x * 2 - 1
		x = self.preprocess(x)
		latents = []
		for i in range(5):
			x = self.encoder[i*2](x)
			latents.append(x)
			x = self.encoder[i*2+1](x)
		del latents[-1]
		latents.reverse()
		return x, latents
	
	def decode_debug(self, z):
		z = self.dec_combiner(self.h.repeat(z.shape[0], 1, 1, 1), z)
		latents = []
		for i in range(4):
			z = self.decoder[i*2](z)
			z = torch.clamp(z, -30, 20)
			z = self.decoder[i*2+1](z)
			latents.append(z)
			zsample = self.sample(z)
			z = self.dec_combiner(z, zsample)
		z = self.postprocess(z)
		z = torch.clamp((z + 1) / 2, 0, 1)
		return z, latents


# In[6]:


stop_time = -1
sample_sides = True

train_once = False
if stop_time >= 0:
	train_once = True
	
load_dict = 1
learn_rate = 1e-4

def rotate_im(x):
	return x.permute(0, 1, 3, 2).contiguous()


# In[7]:


AutoEncoder()


# In[8]:


#odel = AutoEncoder(att).cuda()
model = AutoEncoder().cuda()


# In[9]:


if load_dict == 1:
	print(model.load_state_dict(torch.load('ae_latest_model.pt')))
elif load_dict == 2:
	print(model.load_state_dict(torch.load('ae_latest_model_best.pt')))
elif load_dict % 5 == 0:
	print(model.load_state_dict(torch.load('ae_model_v1_' + str(load_dict) + '.pt')))


# In[10]:


parameters = model.parameters()
optim = torch.optim.Adamax(parameters, lr=learn_rate, betas=(0.6, 0.9), eps=1e-6)
#optim = torch.optim.Adam(parameters, lr=learn_rate, betas=(0.6, 0.9), eps=1e-6)
#optim = torch.optim.SGD(parameters, lr=learn_rate, momentum=0.8)
#optim = torch.optim.AdamW(parameters, lr=learn_rate)
nn.utils.clip_grad_value_(parameters, 1e-2)


# In[11]:


x_best, xre_best, score_best = 0, 0, 100
x_worst, xre_worst, score_worst = 0, 0, -100
train_ims = np.load("ae_train1.npy").transpose(0, 3, 1, 2)
#rain_ims = train_ims * 2 - 1


# In[12]:


dataset = torch.from_numpy(train_ims[30000:40000]).float() / 256. # shape of imgs : (2814, 3, 400, 400)
img_count = train_ims.shape[0]
print(train_ims.shape, dataset.shape)

retry_max_time = 10
retried_times = 0
batch_size = 8


# In[ ]:


#last: AvgLoss=0.00282, AvgRecLoss=0.00262, Loss=0.00258, RecLoss=0.00224
torch.cuda.empty_cache()
x, dataloader = 0, 0
x_rec, rec_loss, otherloss = 0, 0, 0
#scaler = torch.cuda.amp.GradScaler()
for epoch in range(400):
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
	pbar = tqdm.tqdm(dataloader)
	
	loss_epoch = 0
	rec_loss_epoch = 0
	passed_time = 0
	
	#x = dataset[0].unsqueeze(0)
	
	for progress, x in enumerate(pbar):
		optim.zero_grad()
		
		if progress == stop_time:
			break
		
		if x.shape[0] != batch_size:
			continue
			
		x = x.cuda()
		
		#with torch.cuda.amp.autocast():
		#	x_rec, rec_loss, otherloss = model(x)
		#	loss = rec_loss + otherloss * loss_weight
		x_rec, rec_loss, otherloss = model(x)
		loss = rec_loss + otherloss * loss_weight
		
		if math.isnan(loss):
			if retried_times > retry_max_time:
				raise ValueError('fuck you!!!!')
			else:
				retried_times += 1
				optim.zero_grad()
				continue
		retried_times = 0

		#scaler.scale(loss).backward()
		#scaler.step(optim)
		#scaler.update()
		loss.backward()
		optim.step()
		
		passed_time += 1
		loss_epoch += loss.item()
		rec_loss_epoch += rec_loss.item()
		
		if progress % 4 == 0:
			with torch.no_grad():
				avg_loss = loss_epoch / passed_time
				avg_rec_loss = rec_loss_epoch / passed_time
				display_scale = 1
				pbar.set_postfix(AvgRecLoss=avg_rec_loss * display_scale,
								 RecLoss=rec_loss.item() * display_scale,
								 AvgLoss=avg_loss * display_scale,
								 Loss=loss.item() * display_scale)
				
		if sample_sides == True:
			if rec_loss < score_best:
				x_best, xre_best = x.detach().cpu(), x_rec.detach().cpu()
				score_best = rec_loss.item()
				
			if rec_loss > score_worst:
				x_worst, xre_worst = x.detach().cpu(), x_rec.detach().cpu()
				score_worst = rec_loss.item()
				
	if train_once == True:
		print('loss_avg:', loss_epoch / (2814 / batch_size))
		break
	
	if (epoch + 1) % 5 == 0:
		torch.save(model.state_dict(), 'ae_model_v1_' + str(epoch + 1) + '.pt')
	torch.save(model.state_dict(), "ae_latest_model.pt")


# In[13]:


def correctim(recovered):
	if recovered.shape[0] == 3:
		recovered = recovered.transpose(1, 2, 0)
	recovered = recovered.clip(0, 1)
	return recovered

startpos = 100000
for start_pos in range(startpos, startpos + 10):
	testim = train_ims[start_pos:batch_size + start_pos,:,:,:] / 256.
	model = model.cuda()
	with torch.no_grad():
		inputim = torch.from_numpy(testim).float().cuda()
		testim = correctim(testim[0])
		
		model.eval()
		
		if start_pos % 2 == 0:
			inputim = rotate_im(inputim)
			testim = testim.swapaxes(1, 0)
		recovered, _, _ = model(inputim)
		recovered = recovered.cpu()
		
		recovered = correctim(recovered[0].numpy())
		
		model.train()
		
		plt.figure()
		plt.subplot(1, 2, 1)
		plt.imshow(testim)
		plt.subplot(1, 2, 2)
		plt.imshow(recovered)
		plt.show()
		
		plt.imshow(testim)
		plt.show()
		plt.imshow(recovered)
		plt.show()


# In[ ]:


start_pos = 100000
testim = train_ims[start_pos:batch_size + start_pos,:,:,:] / 256.
with torch.no_grad():
		inputim = torch.from_numpy(testim).float().cuda()
		
		model.eval()
		
		mu, log_var = model.encode(inputim)
		z = torch.randn_like(mu) * torch.exp(log_var / 2) + mu
		recovered = model.decode(z)
		recovered = recovered.cpu().numpy()
		
		model.train()
		
		testim = testim.transpose(0, 2, 3, 1)
		recovered = recovered.transpose(0, 2, 3, 1)
		
		plt.imshow(testim[0])
		plt.show()
		plt.imshow(recovered[0])
		plt.show()

