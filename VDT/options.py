"""
@File: options.py
@Time: 2022/11/6
@Author: rp
@Software: PyCharm

"""
import argparse
# RGBD
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')   #5e-5
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='pretrain/swin_base_patch4_window12_384_22k.pth', help='train from checkpoints')
parser.add_argument('--load_pre', type=str, default='./DMTNet_RGBD_cpts/DMTNet_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='3', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='./triple_modal_dataset/Train/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='./triple_modal_dataset/Train/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='./triple_modal_dataset/Train/GT/', help='the training gt images root')
parser.add_argument('--thermal_root', type=str, default='./triple_modal_dataset/Train/thermal/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='./triple_modal_dataset/Train/Edge/', help='the training edge images root')
parser.add_argument('--test_rgb_root', type=str, default='./triple_modal_dataset/Test/RGB/', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str, default='./triple_modal_dataset/Test/depth/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='./triple_modal_dataset/Test/GT/', help='the test gt images root')
parser.add_argument('--test_thermal_root', type=str, default='./triple_modal_dataset/Test/thermal/', help='the test gt images root')
parser.add_argument('--test_edge_root', type=str, default='./triple_modal_dataset/Test/Edge/', help='the test edge images root')
parser.add_argument('--save_path', type=str, default='./checkpoints/new/', help='the path to save models and logs')
opt = parser.parse_args()