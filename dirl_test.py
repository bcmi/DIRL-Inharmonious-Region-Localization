import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import cv2

from data import ihd_dataset
from dirl_train import Trainer
from options import ArgsParser
from data.ihd_dataset import IhdDataset

from evaluation.metrics import MAE, FScore, compute_IoU, normPRED, compute_mAP
from sklearn.metrics import average_precision_score	

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tensor2np(x, isMask=False):
	if isMask:
		if x.shape[1] == 1:
			x = x.repeat(1,3,1,1)
		x = ((x.cpu().detach()))*255
	else:
		x = x.cpu().detach()
		mean = torch.zeros_like(x)
		std = torch.zeros_like(x)
		mean[:,0,:,:] = 0.485
		mean[:,1,:,:] = 0.456
		mean[:,2,:,:] = 0.406
		std[:,0,:,:]  = 0.229
		std[:,1,:,:]  = 0.224
		std[:,2,:,:]  = 0.225
		x = (x * std + mean)*255
		
	return x.numpy().transpose(0,2,3,1).astype(np.uint8)

def save_output(input, mask_label, mask_pred, save_dir, img_fn, extra_infos=None,  verbose=False, alpha=0.5):
	outs = []
	input = cv2.cvtColor(tensor2np(input)[0], cv2.COLOR_RGB2BGR)
	mask_label = tensor2np(mask_label, isMask=True)[0]
	outs += [input, mask_label]
	outs += [tensor2np(v, isMask=True)[0] for k,v in mask_pred.items()]

	outimg = np.concatenate(outs, axis=1) 
	if verbose==True:
		print("show")
		cv2.imshow("out",outimg)
		cv2.waitKey(0)
	else:
		sub_key = os.path.split(img_fn)[1][0]
		if sub_key == 'a': sub_dir = 'adobe'
		if sub_key == 'f': sub_dir = 'flickr'
		if sub_key == 'd': sub_dir = 'day2night'
		if sub_key == 'c': sub_dir = 'coco'
		save_dir = os.path.join(save_dir, sub_dir)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		cv2.imwrite(os.path.join(save_dir, os.path.split(img_fn)[1]), outimg)

# --------- 2. dataloader ---------
#1. dataload
opt = ArgsParser()
test_inharm_dataset = IhdDataset(opt)
test_inharm_dataloader = DataLoader(test_inharm_dataset, batch_size=1,shuffle=False,num_workers=1)

# --------- 3. model define ---------


print("...load DIRLNet...")
checkpoints_dir_root = os.path.split(opt.checkpoints_dir)[0]
model_names = [opt.checkpoints_dir.split('/')[-1]]
prediction_dir = os.path.join(opt.checkpoints_dir, "rst")
if not os.path.exists(prediction_dir): os.makedirs(prediction_dir)
trainers = {}

for name in model_names:
	ckpt_root = os.path.join(checkpoints_dir_root, name)
	opt.checkpoints_dir = ckpt_root
	opt.is_train = 0
	trainer = Trainer(opt)
	trainer.resume(opt.resume, preference=['encoder', 'decoder'])
	trainers[name] = trainer

device = trainers[model_names[0]].device

# trainer.val(is_test=True)
# exit(0)
# ------------ Evaluation Metrics -------------
total_iters = 0
mAP_meter = {name:AverageMeter() for name in model_names}
f1_meter = {name:AverageMeter() for name in model_names}
mIoU_meter = {name:AverageMeter() for name in model_names}

save_flag = True
trainer.encoder.eval()
trainer.decoder.eval()
# --------- 4. inference for each image ---------
for i_test, data in enumerate(test_inharm_dataloader):
	inharmonious, harmonious, mask_gt = data['comp'], data['real'], data['mask']

	inharmonious = inharmonious.type(torch.FloatTensor).to(device)
	harmonious = harmonious.type(torch.FloatTensor).to(device)
	mask_gt = mask_gt.type(torch.FloatTensor).to(device)
	

	with torch.no_grad():
		rsts = {}
		for name in model_names:
			model = trainers[name]
			inharmonious_pred, harmonious_pred = model.forward(inharmonious, harmonious)
			inharmonious_pred, harmonious_pred = inharmonious_pred[0], harmonious_pred[0]

			inharmonious_pred = normPRED(inharmonious_pred)
			mask_gt = normPRED(mask_gt)

			rsts[name] = inharmonious_pred
			pred = inharmonious_pred
			label = mask_gt

			F1 = FScore(pred, label)
			# FBeta = FScore(pred, label, beta2=0.3)
			mAP = compute_mAP(pred, label)
			IoU = compute_IoU(pred, label)
			mIoU_meter[name].update(IoU, mask_gt.size(0))
			mAP_meter[name].update(mAP, mask_gt.size(0))
			f1_meter[name].update(F1, mask_gt.size(0))
		

			if total_iters % 100 == 0:
				print("mAP:\t{}\tF1:\t{}\tmIoU:\t{}".format(mAP_meter[name].avg, f1_meter[name].avg, mIoU_meter[name].avg))

		if save_flag:
			save_output(inharmonious, mask_gt, rsts, prediction_dir, data['img_path'][0], extra_infos=None, verbose=False)


for name in model_names:
	print("Model:\t{}".format(name))
	print("Average mAP:\t{}".format(mAP_meter[name].avg))
	print("Average F1 Score:\t{}".format(f1_meter[name].avg))
	print("Average IoU:\t{}".format(mIoU_meter[name].avg))		
		
		
