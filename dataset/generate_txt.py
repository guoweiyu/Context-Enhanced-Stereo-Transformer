import os
from natsort import natsorted
from utilities.python_pfm import readPFM
from utilities.misc import find_occ_mask
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import utils as vutils

datadir ="/data/middlebury2014_full/"
disp_left_fname='disp0.pfm'
disp_right_fname='disp1.pfm'
occ_left_fname='mask0nocc.pth'
occ_right_fname='mask1nocc.pth'

occ=torch.load("/data/middlebury2014_full/Bicycle1-imperfect/mask0nocc.pth")
plt.figure(1)
plt.imshow(occ,cmap='gray')
plt.show()
occ_left = np.array(occ) == 128
plt.figure(2)
plt.imshow(occ_left,cmap='gray')
plt.show()

occ_left_data=[os.path.join(obj, occ_left_fname) for obj in
                               os.listdir(os.path.join(datadir))]
occ_right_data=[os.path.join(obj, occ_right_fname) for obj in
                               os.listdir(os.path.join(datadir))]
disp_left_data = [os.path.join(obj, disp_left_fname) for obj in
                               os.listdir(os.path.join(datadir))]
disp_right_data = [os.path.join(obj, disp_right_fname) for obj in
                               os.listdir(os.path.join(datadir))]
disp_left_data = natsorted(disp_left_data)
disp_right_data = natsorted(disp_right_data)
occ_left_data=natsorted(occ_left_data)
occ_right_data=natsorted(occ_right_data)

for idx in range(45):
    disp_left_fname1 = os.path.join(datadir, disp_left_data[idx])
    disp_right_fname1 = os.path.join(datadir, disp_right_data[idx])
    occ_l_fname1 =os.path.join(datadir, occ_left_data[idx])
    occ_r_fname1 = os.path.join(datadir, occ_right_data[idx])
    disp_left, _ = readPFM(disp_left_fname1)
    disp_right, _ = readPFM(disp_right_fname1)

    occ_mask_l, occ_mask_r=find_occ_mask(disp_left,disp_right)
    print(occ_l_fname1)
    occ_mask_l=torch.tensor(occ_mask_l)
    occ_mask_r = torch.tensor(occ_mask_r)
    torch.save(occ_mask_l,occ_l_fname1)
    torch.save(occ_mask_r, occ_r_fname1)
    # plt.imshow(occ_mask_l)
    # plt.savefig(occ_l_fname1)

