

import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from natsort import natsorted
import random
from dataset.preprocess import augment, normalization
from dataset.stereo_albumentation import random_crop, horizontal_flip
from utilities.python_pfm import readPFM


class MiddleburyBaseDataset(data.Dataset):
    def __init__(self, datadir, split='train'):
        super(MiddleburyBaseDataset, self).__init__()

        self.datadir = datadir
        self.split = split

        self.left_fname = None
        self.right_fname = None
        self.disp_left_fname = None
        self.disp_right_fname = None
        self.occ_left_fname = None
        self.occ_right_fname = None

        self._augmentation()

    def _read_data(self):
        self.left_data = [os.path.join(obj, self.left_fname) for obj in os.listdir(os.path.join(self.datadir))]
        self.right_data = [os.path.join(obj, self.right_fname) for obj in os.listdir(os.path.join(self.datadir))]
        self.disp_left_data = [os.path.join(obj, self.disp_left_fname) for obj in
                               os.listdir(os.path.join(self.datadir))]
        #self.disp_right_data = [os.path.join(obj, self.disp_right_fname) for obj in
        #                        os.listdir(os.path.join(self.datadir))]
        self.occ_left_data = [os.path.join(obj, self.occ_left_fname) for obj in
                              os.listdir(os.path.join(self.datadir))]
        #self.occ_right_data = [os.path.join(obj, self.occ_right_fname) for obj in
        #                       os.listdir(os.path.join(self.datadir))]
        self.left_data = natsorted(self.left_data)
        self.right_data = natsorted(self.right_data)
        self.disp_left_data = natsorted(self.disp_left_data)
        #self.disp_right_data = natsorted(self.disp_right_data)
        self.occ_left_data = natsorted(self.occ_left_data)
        #self.occ_right_data = natsorted(self.occ_right_data)

    def _augmentation(self):
        self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        input_data = {}

        # left
        left_fname = os.path.join(self.datadir, self.left_data[idx])
        left = np.array(Image.open(left_fname)).astype(np.uint8)
        input_data['left'] = left

        # right
        right_fname = os.path.join(self.datadir, self.right_data[idx])
        right = np.array(Image.open(right_fname)).astype(np.uint8)
        input_data['right'] = right

        if not self.split == 'test':  # no disp for test files
            # occ
            occ_left_fname = os.path.join(self.datadir, self.occ_left_data[idx])
            #occ_right_fname = os.path.join(self.datadir, self.occ_right_data[idx])
            occ_left = np.array(Image.open(occ_left_fname)) == 128
            #occ_right = np.array(Image.open(occ_right_fname)) == 128

            # disp
            disp_left_fname = os.path.join(self.datadir, self.disp_left_data[idx])
            #disp_right_fname = os.path.join(self.datadir, self.disp_right_data[idx])

            disp_left, _ = readPFM(disp_left_fname)
            #disp_right, _ = readPFM(disp_right_fname)

            if self.split == 'train':
                input_data['disp'] = disp_left
                input_data['occ_mask'] = occ_left
                # horizontal flip
                # input_data['left'], input_data['right'], input_data['occ_mask'], input_data['occ_mask_right'], \
                # input_data['disp'],\
                # input_data['disp_right'] = horizontal_flip(input_data['left'], input_data['right'], occ_left, occ_right,
                #                                            disp_left,
                #                                             self.split)
                # random crop

                input_data = random_crop(648, 1000, input_data, self.split)


            else:

                input_data['occ_mask'] = occ_left
                #input_data['occ_mask_right'] = occ_right
                input_data['disp'] = disp_left

                #input_data['disp_right'] = disp_right
            input_data = augment(input_data, self.transformation)
        else:
            input_data = normalization(**input_data)

        return input_data

class MiddleburyBaseDataset1(data.Dataset):
    def __init__(self, datadir, split='train'):
        super(MiddleburyBaseDataset1, self).__init__()

        self.datadir = datadir
        self.split = split


        self.left_fname = None
        self.right_fname = None
        #self.right_fname_list=['im1.png','im1E.png','im1L.png']
        self.right_fname_list = ['im1.png']
        self.disp_left_fname = None
        self.disp_right_fname = None
        self.occ_left_fname = None
        self.occ_right_fname = None
        self.scenes = None
        self._augmentation()

    def _read_data(self):
        self.scenes = [obj for obj in os.listdir(os.path.join(self.datadir))]
        self.disp_left_data = [os.path.join(obj, self.disp_left_fname) for obj in
                               os.listdir(os.path.join(self.datadir))]



        #self.disp_right_data = [os.path.join(obj, self.disp_right_fname) for obj in
        #                        os.listdir(os.path.join(self.datadir))]
        self.occ_left_data = [os.path.join(obj, self.occ_left_fname) for obj in
                              os.listdir(os.path.join(self.datadir))]

        #self.occ_right_data = [os.path.join(obj, self.occ_right_fname) for obj in
        #                       os.listdir(os.path.join(self.datadir))]
        self.scenes= natsorted(self.scenes)

        self.disp_left_data = natsorted(self.disp_left_data)
        self.occ_left_data = natsorted(self.occ_left_data)

        #self.occ_right_data = natsorted(self.occ_right_data)


    def _augmentation(self):
        self.transformation = None

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        input_data = {}
        downrate=4
        # left

        left_fname = os.path.join(self.datadir, self.scenes[idx],self.left_fname)
        #print(left_fname)
        left = np.array(Image.open(left_fname)).astype(np.uint8)
        input_data['left'] = left[::downrate,::downrate,:]

        # right
        random.shuffle(self.right_fname_list)
        right_fname = os.path.join(self.datadir,self.scenes[idx], self.right_fname_list[0])
        #print(right_fname)
        right = np.array(Image.open(right_fname)).astype(np.uint8)

        input_data['right'] = right[::downrate,::downrate,:]

        if not self.split == 'test':  # no disp for test files
            # occ
            occ_left_fname = os.path.join(self.datadir, self.occ_left_data[idx])
            #print(occ_left_fname)

            occ_left = np.array(torch.load(occ_left_fname)) == 128


            # disp
            disp_left_fname = os.path.join(self.datadir, self.disp_left_data[idx])
            #print(disp_left_fname)
            #disp_right_fname = os.path.join(self.datadir, self.disp_right_data[idx])

            disp_left, _ = readPFM(disp_left_fname)
            #disp_right, _ = readPFM(disp_right_fname)
            down_disp = disp_left[::downrate, ::downrate]
            down_disp = down_disp / (disp_left.shape[1] / down_disp.shape[1])
            if self.split == 'train':

                input_data['disp'] = down_disp
                input_data['occ_mask'] = occ_left[::downrate,::downrate]
                # horizontal flip
                # input_data['left'], input_data['right'], input_data['occ_mask'], input_data['occ_mask_right'], \
                # input_data['disp'],\
                # input_data['disp_right'] = horizontal_flip(input_data['left'], input_data['right'], occ_left, occ_right,
                #                                            disp_left,
                #                                             self.split)
                # random crop

                input_data = random_crop(384, 1000, input_data, self.split)


            else:

                input_data['occ_mask'] = occ_left[::downrate,::downrate]
                #input_data['occ_mask_right'] = occ_right
                input_data['disp'] = down_disp

                #input_data['disp_right'] = disp_right
            input_data = augment(input_data, self.transformation)
        else:
            input_data = normalization(**input_data)

        return input_data
class Middlebury2014Dataset(MiddleburyBaseDataset):
    def __init__(self, datadir, split='train'):
        super(Middlebury2014Dataset, self).__init__(datadir, split)
        self.left_fname = 'im0.png'
        self.right_fname = 'im1.png'
        self.disp_left_fname = 'disp0GT.pfm'
        #self.disp_right_fname = 'disp1GT.pfm'
        self.occ_left_fname = 'mask0nocc.png'
        #self.occ_right_fname = 'mask1nocc.png'

        self._read_data()

class Middlebury2014Dataset1(MiddleburyBaseDataset1):
    def __init__(self, datadir, split='train'):
        super(Middlebury2014Dataset1, self).__init__(datadir, split)
        self.left_fname = 'im0.png'
        self.right_fname = 'im1.png'
        self.disp_left_fname = 'disp0.pfm'
        #self.disp_right_fname = 'disp1GT.pfm'
        self.occ_left_fname = 'mask0nocc.png'
        #self.occ_right_fname = 'mask1nocc.png'

        self._read_data()
