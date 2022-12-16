import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import vgg16
import pickle

from sklearn.model_selection import KFold
from data_loaders.lol_v1_new import lowlight_loader_new
from model.IAT_main import IAT

from IQA_pytorch import SSIM
from utils import PSNR, adjust_learning_rate, validation, LossNetwork, visualization

import madgrad

from msssim import ms_ssim

def ms_ssim_l1(y_pred, y_true):
    #Gaussian filter (dot) l1_loss
    #F.L1_loss(y_pred, y_true)

    l1 = torch.sum(torch.absolute(y_pred - y_true)) / np.prod(y_pred.shape)
    
    ms_ssim_loss = 1 - ms_ssim(enhance_img, high_img, data_range=torch.max(enhance_img))
    
    #return ms_ssim_loss
    return 0.84 * ms_ssim_loss + (1 - 0.84) * l1

save_dir = "/content/drive/MyDrive/Project_B/Training/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--img_path', type=str, default='./dataset/low/')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0004)

    parser.add_argument('--num_epochs', type=int, default=10)

    config = parser.parse_args()

    if config.name is None:
        print("Model save name MUST be passed with --name <name>")
        exit()
        

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    os.makedirs(save_dir + config.name + "/model_training/", exist_ok=True)
    os.makedirs(save_dir + config.name + "/model_validation/", exist_ok=True)
    # Model Setting
    model = IAT().cuda()

    # L1_loss = CharbonnierLoss()
    #L1Loss is MAE. Wtf.
    #L1_loss = nn.L1Loss()
    #loss_function = nn.MSELoss()
    loss_function = nn.L1Loss()
    #loss_function = ms_ssim_l1
    L1_smooth_loss = F.smooth_l1_loss

    ssim = SSIM()
    psnr = PSNR()
    ssim_high = 0
    psnr_high = 0
    
    num_kfolds = 5

    training_scores = {'loss': np.zeros((num_kfolds, config.num_epochs))}
    validation_scores = {'loss': np.zeros((num_kfolds, config.num_epochs)), 'PSNR': np.zeros((num_kfolds, config.num_epochs)), 'SSIM': np.zeros((num_kfolds, config.num_epochs))}

    kf = KFold(n_splits=num_kfolds, shuffle=True)
    
    ids = np.arange(0, 500, 1)
    for k, (train_index, val_index) in enumerate(kf.split(ids)):
        train_ids = ids[train_index]
        val_ids = ids[val_index]

        use_clahe = False
        # Data Setting
        train_dataset = lowlight_loader_new(images_path=config.img_path, ids = train_ids, use_clahe=use_clahe)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                                                   pin_memory=True)
        val_dataset = lowlight_loader_new(images_path=config.img_path, mode='test', ids = val_ids, use_clahe=use_clahe)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # Model Setting
        model = IAT().cuda()

        #Madgrad?
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
        #optimizer = madgrad.MADGRAD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

        device = next(model.parameters()).device
        print('the device is:', device)

        print('######## Start IAT Training #########')
        for epoch in range(0, config.num_epochs):
            # adjust_learning_rate(optimizer, epoch)
            print('\nThe epoch is:', epoch + 1)
            
            epoch_loss = []
            for iteration, imgs in enumerate(train_loader):
                low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
                batch_filenames = imgs[2]

                optimizer.zero_grad()
                model.train()
                mul, add, enhance_img = model(low_img)
                
                for i in range(0, low_img.shape[0]):
                    x1 = enhance_img[i].cpu().detach().numpy()
                    x2 = high_img[i].cpu().detach().numpy()
                    
                    x1 = np.transpose(x1, (1, 2, 0))
                    x2 = np.transpose(x2, (1, 2, 0))
                    
                    output_img = np.zeros((x1.shape[0], 1 + x1.shape[1] + 1 + x2.shape[1] + 1, 3)).astype('uint8')
                    
                    output_img[:, 1: x1.shape[1] + 1, :] = (x1 * 255).astype("uint8")
                    output_img[:, 1 + x1.shape[1] + 2:, :] = (x2 * 255).astype("uint8")
                    
                    cv2.imwrite(save_dir + config.name + "/model_training/" + batch_filenames[i] + ".png", cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
                
                #loss = ms_ssim_l1(enhance_img, high_img)
                loss = loss_function(enhance_img, high_img)
                #loss = MS_SSIM_l1(enhance_img, high_img)
                #loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
                loss.backward()

                optimizer.step()
                scheduler.step()

                print("\tLoss at batch", iteration + 1, ":", loss.item(), end='\r')

                epoch_loss.append(loss.item())
            
            training_scores['loss'][k][epoch] = np.average(epoch_loss)
        
            print('\n')
            # Evaluation Model
            model.eval()
            SSIM_mean, PSNR_mean, loss_mean = validation(model, val_loader, ms_ssim_l1)

            validation_scores['PSNR'][k][epoch] = PSNR_mean
            validation_scores['SSIM'][k][epoch] = SSIM_mean
            validation_scores['loss'][k][epoch] = loss_mean
            
            if SSIM_mean > ssim_high:
                ssim_high = SSIM_mean
                print('\tThe highest SSIM value is:', str(ssim_high))
                torch.save(model.state_dict(), os.path.join(save_dir + config.name + "/" + "fold_%d_best_Epoch.pth" % k))

    
    history = {'training': training_scores, 'validation': validation_scores}
    pickle.dump(history, open(save_dir + config.name + "/training_history.pickle", "wb"))
    
