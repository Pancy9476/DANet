from models.net import Encoder, Decoder, iv_fuse_sche
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import utils.logger as Logger
import warnings
import models as Model
import argparse
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/diffusion.json', help='configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='testing', default='test')
args = parser.parse_args()
opt = Logger.parse(args)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = r"weights/danet.pth"
Encoder = nn.DataParallel(Encoder()).to(device)
Decoder = nn.DataParallel(Decoder()).to(device)
Encoder.load_state_dict(torch.load(ckpt_path)['Encoder'])
Decoder.load_state_dict(torch.load(ckpt_path)['Decoder'])

for dataset_name in ["TNO", "RoadScene", "MSRS"]:
    test_folder = os.path.join('test_img', dataset_name)
    result_folder = os.path.join('test_result', dataset_name)
    Encoder.eval()
    Decoder.eval()
    diffusion = Model.create_model(opt)
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
            print(img_name)
            img_ir = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            img_vi = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            img_ir, img_vi = torch.FloatTensor(img_ir), torch.FloatTensor(img_vi)
            img_vi, img_ir = img_vi.cuda(), img_ir.cuda()
            feature_V_D, feature_V = Encoder(img_vi)
            feature_I_D, feature_I = Encoder(img_ir)
            img_fuse = iv_fuse_sche(sche='sum', f_v=feature_V_D, f_i=feature_I_D)
            diff_fuse = torch.cat((img_vi, img_ir), dim=1)
            diffusion.feed_data(diff_fuse)
            latent_f = torch.zeros(1, 64, len(diff_fuse[0][0]), len(diff_fuse[0][0][0])).to('cuda')
            for t in [5, 50, 100]:
                e_t, d_t = diffusion.get_feats(t=t)
                latent_f += d_t
            latent_f /= len([5, 50, 100])
            img_fuse = img_fuse + latent_f
            img_Fuse = Decoder(img_vi, img_fuse)
            img_Fuse = (img_Fuse - torch.min(img_Fuse)) / (torch.max(img_Fuse) - torch.min(img_Fuse))
            fi = np.squeeze((img_Fuse * 255).cpu().numpy())
            img_save(fi, img_name.split(sep='.')[0], result_folder)