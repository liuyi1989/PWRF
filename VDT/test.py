"""
@File: CatNet_test.py
@Time: 2022/11/6
@Author: rp
@Software: PyCharm

"""
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.DMTNet import DMTNet
# from cpts4.Swin_Transformer import SwinNet
from tools.data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='2', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/lcx/DMTNet/triple_modal_dataset/D-challenge/',help='test dataset path')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#dataset_path ='/home/lcx/DMTNet/triple_modal_dataset/T-challenge/'  #D-challenge
dataset_path ='/home/lcx/CATNet-main/fea_dataset/'  #D-challenge

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = DMTNet()
model.load_state_dict(torch.load('./checkpoints/new/DMTNet_epoch_best.pth'))
model.cuda()
model.eval()
# test


test_datasets = ['val']

#test_datasets = ['Test']
#test_datasets = ['V-BSO','V-LI','V-MSO','V-NI','V-SA','V-SI','V-SSO']

#test_datasets = ['D-BI','D-BM','D-II', 'D-SSO']
#test_datasets = ['T-Cr','T-HR', 'T-RD']


for dataset in test_datasets:
    save_path = './final/train_VDT/' + dataset + '/'
    edge_save_path = './final/VDT/edge/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(edge_save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    thermal_root = dataset_path + dataset + '/thermal/'
    test_loader = test_dataset(image_root, gt_root, depth_root, thermal_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, thermal, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth = depth.repeat(1,3,1,1).cuda()
        thermal = thermal.cuda()
        res_, edge,res1,_,_,res= model(image,depth,thermal)
  
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        edge = F.upsample(edge, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        print('save img to: ',save_path+name)
        # ndarray to image
        cv2.imwrite(save_path + name, res*255)
        cv2.imwrite(edge_save_path + name, edge * 255)
    print('Test Done!')
