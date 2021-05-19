import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
import numpy as np
import glob
import os
import itertools
import cv2

import multiprocessing as mp


from model import InharmoniousDecoder, InharmoniousEncoder
from evaluation.metrics import FScore, normPRED, compute_mAP, compute_IoU



from data.ihd_dataset import IhdDataset
from options import ArgsParser
import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target, is_harmonious=False, loss_mode=''):
    bce_out = bce_loss(pred,target)
    if is_harmonious:
        ssim_out = 0
        if loss_mode == '':
            iou_out = 0
    else:
        ssim_out = 1 - ssim_loss(pred,target)
        iou_out = iou_loss(pred,target)
    if loss_mode == '':
        loss = bce_out + ssim_out + iou_out
        return {"total":loss, "bce":bce_out, "ssim":ssim_out, "iou":iou_out}
    else:
        loss = bce_out + ssim_out
        return {"total":loss, "bce":bce_out, "ssim":ssim_out}
    
def multi_bce_loss_fusion(preds, labels_v, weights=1, is_harmonious=False, loss_mode=''):
    total_loss = 0
    bce_out = 0
    ssim_out = 0
    if isinstance(weights, int):
        weights = [weights] * len(preds)
    if loss_mode == '':
        iou_out = 0
    for pred,w in zip(preds,weights):
        loss = bce_ssim_loss(pred, labels_v, is_harmonious, loss_mode)
        total_loss += loss['total'] * w
        bce_out += loss['bce']
        ssim_out += loss['ssim']
        if loss_mode == '':
            iou_out += loss['iou']
    if loss_mode == '':
        return {"total":total_loss, "bce":bce_out, "ssim":ssim_out, "iou":iou_out}
    else:
        return {"total":total_loss, "bce":bce_out, "ssim":ssim_out}

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

class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        if opt.is_train:
            log_dir = os.path.join(opt.checkpoints_dir, "logs")
            if not os.path.exists(log_dir): os.makedirs(log_dir)
            self.writer = SummaryWriter(log_dir)   # create a visualizer that display/save images and plots
        self.gpus = opt.gpu_ids.split(',')
        self.gpus = [int(id) for id in self.gpus]
        self.device = torch.device('cuda:{}'.format(self.gpus[0])) if self.gpus[0]>-1 else torch.device('cpu')  # get device name: CPU or GPU
        print(self.device)

        self.best_acc = 0
        
        # ------- 3. define model --------
        self.encoder = InharmoniousEncoder(opt, opt.input_nc)
        self.decoder = InharmoniousDecoder(opt, opt.input_nc)
        
        encoder_size = sum(p.numel() for p in self.encoder.parameters())/1e6
        decoder_size = sum(p.numel() for p in self.decoder.parameters())/1e6
        print('--- Encoder params: %.2fM' % (encoder_size))
        print('--- Total params: %.2fM' % (encoder_size+decoder_size))
        if len(self.gpus) > 1:
            if opt.sync_bn:
                print("Using distributed training !")
                mp.set_start_method('forkserver')
                self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
                self.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder)
                torch.distributed.init_process_group(backend="nccl", rank=0, world_size=len(self.gpus),init_method=opt.port)
                self.dataparallel_func = nn.parallel.DistributedDataParallel
            else:
                self.dataparallel_func = nn.DataParallel
        else:
            self.dataparallel_func = None
       
        if opt.is_train == 1:
            if self.dataparallel_func is not None:
                self.decoder = self.dataparallel_func(self.decoder.to(self.device), self.gpus)
                self.encoder = self.dataparallel_func(self.encoder.to(self.device), self.gpus)

            else:
                self.encoder.to(self.device)
                self.decoder.to(self.device)

        else:
            self.encoder = self.encoder.to(self.device)
            self.encoder.eval()
            self.decoder = self.decoder.to(self.device)
            self.decoder.eval()

        # ------- 2. set the directory of training dataset --------
        self.data_mean = opt.mean.split(",")
        self.data_mean = [float(m.strip()) for m in self.data_mean]
        self.data_std = opt.std.split(",")
        self.data_std = [float(m.strip()) for m in self.data_std]
        
        opt.phase = 'train'
        inharm_dataset = IhdDataset(opt)
        if opt.is_train == 0:
            opt.batch_size = 1
            opt.num_threads = 1
            opt.serial_batches = True
        if not opt.sync_bn:
            self.inharm_dataloader = torch.utils.data.DataLoader(
                        inharm_dataset,
                        batch_size=opt.batch_size,
                        shuffle=not opt.serial_batches,
                        num_workers=int(opt.num_threads),
                        drop_last=True)
        else:
            self.inharm_dataloader = torch.utils.data.DataLoader(
                inharm_dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads),
                sampler=torch.utils.data.distributed.DistributedSampler(inharm_dataset)
            )
        opt.phase = 'val'
        opt.preprocess = 'resize'
        opt.no_flip = True
        self.val_dataloader = torch.utils.data.DataLoader(
                    IhdDataset(opt),
                    batch_size=1,
                    shuffle=False,
                    num_workers=1)
        opt.is_train = True
        # ------- 4. define optimizer --------
        if opt.is_train:
            self.image_display = None
            print("---define optimizer...")
            self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(0.9,0.999), weight_decay=opt.weight_decay)
            self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
            self.decoder_schedular   = optim.lr_scheduler.MultiStepLR(self.decoder_opt, milestones=[30, 40, 50, 55], gamma=0.5)
            self.encoder_schedular   = optim.lr_scheduler.MultiStepLR(self.encoder_opt, milestones=[30, 40, 50, 55], gamma=0.5)
            
    def adjust_learning_rate(self):
        self.encoder_schedular.step()
        self.decoder_schedular.step()

    def write_display(self, total_it, model, batch_size):
        # write loss
        members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and attr.startswith('loss')]
        for m in members:
            self.writer.add_scalar(m, getattr(model, m), total_it)
        # write img
        if isinstance(model.image_display, torch.Tensor):
            image_dis = torchvision.utils.make_grid(model.image_display, nrow=batch_size)
            mean = torch.zeros_like(image_dis)
            mean[0,:,:] = .485
            mean[1,:,:] = .456
            mean[2,:,:] = .406
            std = torch.zeros_like(image_dis)
            std[0,:,:] = 0.229
            std[1,:,:] = 0.224
            std[2,:,:] = 0.225
            image_dis = image_dis*std + mean
            self.writer.add_image('Image', image_dis, total_it)

    def load_dict(self, net, name, resume_epoch, strict=True):
        ckpt_name = "{}_epoch{}.pth".format(name, resume_epoch)
        if not os.path.exists(os.path.join(self.opt.checkpoints_dir, ckpt_name)): 
            if resume_epoch == -2:
                ckpt_name = "{}_epoch{}.pth".format(name, "best")
            else:
                ckpt_name = "{}_epoch{}.pth".format(name, "latest")
        print("Loading model weights from {}...".format(ckpt_name))

        # restore lr
        sch = getattr(self, '{}_schedular'.format(name))
        sch.last_epoch = resume_epoch if resume_epoch > 0 else 0
        decay_coef = 0
        for ms in sch.milestones.keys():
            if sch.last_epoch <= ms: decay_coef+=1

        for group in sch.optimizer.param_groups:
            group['lr'] = group['lr'] * sch.gamma ** decay_coef
        
        ckpt_dict = torch.load(os.path.join(self.opt.checkpoints_dir,ckpt_name), map_location=self.device)
        if 'best_acc' in ckpt_dict.keys():
            new_state_dict = ckpt_dict['state_dict']
            save_epoch = ckpt_dict['epoch']
            self.best_acc  = ckpt_dict['best_acc']
            print("The model from epoch {} reaches acc at {:.4f} !".format(save_epoch, self.best_acc))
        else:
            new_state_dict = ckpt_dict
            
        current_state_dict = net.state_dict()
        new_keys = tuple(new_state_dict.keys())
        for k in new_keys:
            if k.startswith('module'):
                v = new_state_dict.pop(k)
                nk = k.split('module.')[-1]
                new_state_dict[nk] = v
        if len(self.gpus) > 1:
            net.module.load_state_dict(new_state_dict, strict=strict)
        else:
            net.load_state_dict(new_state_dict, strict=True) # strict


    def resume(self, resume_epoch, strict=True, is_pretrain=False, preference=[]):
        if preference != []:
            for net_name in preference:
                net = getattr(self, net_name)
                self.load_dict(net, net_name, resume_epoch, strict=strict)
            return 

    def save(self, epoch, is_pretrain=False, preference=[]):
        if preference != []:
            for net_name in preference:
                model_name = "{}_epoch{}.pth".format(net_name, epoch)
                net = getattr(self, net_name)
                save_dict = {
                    'epoch':epoch,
                    'best_acc':self.best_acc,
                    'state_dict':net.state_dict(),
                    'opt':getattr(self, '{}_schedular'.format(net_name)).state_dict()
                }
                torch.save(save_dict, os.path.join(self.opt.checkpoints_dir, model_name))
            return 

    def denormalize(self, x, isMask=False):
        if isMask:
            mean = 0
            std=1
        else:
            mean = torch.zeros_like(x)
            mean[:,0,:,:] = .485
            mean[:,1,:,:] = .456
            mean[:,2,:,:] = .406
            std = torch.zeros_like(x)
            std[:,0,:,:] = 0.229
            std[:,1,:,:] = 0.224
            std[:,2,:,:] = 0.225
        x = (x*std + mean)*255
        x = x.cpu().detach().numpy().transpose(0,2,3,1).astype(np.uint8)
        if isMask:
            x = x.repeat(3, axis=3)
        return x


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, inharmonious, harmonious):
        z_inharm = self.encoder(inharmonious)
        z_harm = self.encoder(harmonious)
        
        inharmonious_outs =self.decoder(z_inharm)
        harmonious_outs =self.decoder(z_harm)

        inharmonious_pred = inharmonious_outs['mask']
        harmonious_pred = harmonious_outs['mask']

        return inharmonious_pred, harmonious_pred
    
    def val(self, is_test=False):
        print("---start validation---")
        total_iters = 0
        mAPMeter = AverageMeter()
        F1Meter = AverageMeter()
        FbMeter = AverageMeter()
        IoUMeter = AverageMeter()
        self.encoder.eval()
        self.decoder.eval()
        for i_test, data in enumerate(self.val_dataloader):
            inharmonious, harmonious, mask_gt = data['comp'], data['real'], data['mask']

            inharmonious = inharmonious.type(torch.FloatTensor).to(self.device)
            harmonious = harmonious.type(torch.FloatTensor).to(self.device)
            mask_gt = mask_gt.type(torch.FloatTensor).to(self.device)            
            with torch.no_grad():
                inharmonious_pred, harmonious_pred = self.forward(inharmonious, harmonious)
                inharmonious_pred, harmonious_pred = inharmonious_pred[0], harmonious_pred[0]

                inharmonious_pred = normPRED(inharmonious_pred)
                mask_gt = normPRED(mask_gt)

                pred = inharmonious_pred
                label = mask_gt

                F1 = FScore(pred, label)
                FBeta = FScore(pred, label, threshold=-1, beta2=0.3)
                
                mAP = compute_mAP(pred, label)

                IoUMeter.update(compute_IoU(pred, label), label.size(0))
                mAPMeter.update(mAP, inharmonious_pred.size(0))
                F1Meter.update(F1, inharmonious_pred.size(0))
                FbMeter.update(FBeta, inharmonious_pred.size(0))

                total_iters += 1
                if total_iters % 100 == 0:
                    print("Batch: [{}/{}],\tmAP:\t{:.4f}\tF1:\t{:.4f}\tFbeta:\t{:.4f}\tIoU:\t{:.4f}".format((i_test+1) , len(self.val_dataloader), \
                        mAPMeter.avg, F1Meter.avg, FbMeter.avg, IoUMeter.avg))
        if is_test:
            name = self.opt.checkpoints_dir.split('/')[-1]
            print("Model\t{}:\nmAP:\t{:.4f}\nF1:\t{:.4f}\nFbeta:\t{:.4f}\nIoU:\t{:.4f}".format(name,\
                        mAPMeter.avg, F1Meter.avg, FbMeter.avg, IoUMeter.avg))
        else:
            val_mIoU = IoUMeter.avg
            if self.best_acc < val_mIoU:
                self.best_acc = val_mIoU
                self.save("best", preference=['encoder','decoder'])
                print("New Best score!\nmAP:\t{:.4f},\tF1:\t{:.4f},\tIoU:\t{:.4f}".format(mAPMeter.avg, F1Meter.avg, val_mIoU))
        
        self.encoder.train()
        self.decoder.train()

    def train(self, start_epoch=0):
        # ------- 5. training process --------
        print("---start training...")
        total_iters = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
        
        loss_det_meter = AverageMeter()
        loss_att_meter = AverageMeter()
        F1Meter = AverageMeter()
        for epoch in range(start_epoch, self.opt.nepochs):
            for i, data in enumerate(self.inharm_dataloader):
                total_iters = total_iters + 1
                ite_num4val = ite_num4val + 1

                inharmonious, harmonious, mask_gt = data['comp'], data['real'], data['mask']

                inharmonious = inharmonious.type(torch.FloatTensor).to(self.device)
                harmonious = harmonious.type(torch.FloatTensor).to(self.device)
                mask_gt = mask_gt.type(torch.FloatTensor).to(self.device)
                 
                inharm_out = self.encoder(inharmonious)
                inharmonious_pred = self.decoder(inharm_out)['mask']
            
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                
                # if use mask supervision
                if self.opt.mda_mode != 'vanilla':
                    loss_inharmonious = multi_bce_loss_fusion([inharmonious_pred[0]], mask_gt, loss_mode=self.opt.loss_mode)
                    self.loss_attention = multi_bce_loss_fusion(inharmonious_pred[1:], mask_gt, loss_mode=self.opt.loss_mode)['total']
                else:
                    loss_inharmonious = multi_bce_loss_fusion(inharmonious_pred, mask_gt,  loss_mode=self.opt.loss_mode)
            
                # self.loss_detection_ssim = loss_inharmonious['ssim']
                self.loss_detection_bce = loss_inharmonious['bce']
                self.loss_detection =  loss_inharmonious['total']
                
                if self.opt.mda_mode != 'vanilla':
                    loss_det_meter.update(self.loss_detection.item(), n=inharmonious.shape[0])
                    loss_att_meter.update(self.loss_attention.item(), n=inharmonious.shape[0])
                    self.loss_total = self.loss_detection  * self.opt.lambda_detection + self.loss_attention * self.opt.lambda_attention
                else:
                    self.loss_total = self.loss_detection * self.opt.lambda_detection

                self.loss_total.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()
               
                # # print statistics
                running_loss += self.loss_total.item()
                F1Meter.update(FScore(inharmonious_pred[0], mask_gt), n=inharmonious_pred[0].size(0))
                if total_iters % self.opt.print_freq == 0:              
                    if self.opt.mda_mode != 'vanilla':
                        print("Epoch: [%d/%d], Batch: [%d/%d], train loss: %.3f, det loss: %.3f, att loss: %.3f, F1 score: %.4f" % (
                        epoch + 1, self.opt.nepochs, (i + 1) , len(self.inharm_dataloader),
                        running_loss / ite_num4val,
                        loss_det_meter.avg,
                        loss_att_meter.avg,
                        F1Meter.avg
                        ))
                    else:
                        print("Epoch: [%d/%d], Batch: [%d/%d], train loss: %.3f, det loss: %.3f, F1 score: %.4f" % (
                        epoch + 1, self.opt.nepochs, (i + 1) , len(self.inharm_dataloader),
                        running_loss / ite_num4val,
                        self.loss_detection.item(),
                        F1Meter.avg
                        ))
                if total_iters %  self.opt.display_freq== 0: #
                    show_size = 5 if inharmonious.shape[0] > 5 else inharmonious.shape[0]
                    self.image_display = torch.cat([
                                    inharmonious[0:show_size].detach().cpu(),             # input image
                                    mask_gt[0:show_size].detach().cpu().repeat(1,3,1,1),                        # ground truth
                                    inharmonious_pred[0][0:show_size].detach().cpu().repeat(1,3,1,1),       # refine out
                    ],dim=0)
                    self.write_display(total_iters, self, show_size)
                    
                # del temporary outputs and loss
                del inharmonious_pred

            self.adjust_learning_rate()
            F1Meter.reset()
            loss_att_meter.reset()
            loss_det_meter.reset()
            running_loss = 0.0
            running_tar_loss = 0.0
            ite_num4val = 0
            self.save("latest", preference=['encoder','decoder'])
            if (epoch+1) % self.opt.save_epoch_freq == 0:  # save model every 2000 iterations
                self.save(epoch+1, preference=['encoder','decoder'])
            if (epoch+1) < 30:
                if (epoch+1) % self.opt.save_epoch_freq == 0:
                    self.val()
            else:
                if (epoch+1) % 3 == 0:
                    self.val()
            
        print('-------------Congratulations, No Errors!!!-------------')

if __name__ == '__main__':
    opt = ArgsParser()
    opt.seed = 42
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
    print(opt.checkpoints_dir.split('/')[-1])
    
    trainer = Trainer(opt)
    start_epoch = 0
    if opt.resume > -1:
        trainer.resume(opt.resume, preference=['encoder', 'decoder'])
        # trainer.resume(opt.resume)
        start_epoch = opt.resume
    trainer.train(start_epoch=start_epoch)
    

