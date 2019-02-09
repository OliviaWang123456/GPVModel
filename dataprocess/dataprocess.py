import os
import numpy as np
import torch
import torch.utils.data as Data
import h5py
import json
from split import DatasetSplit
import skimage.io as sio
import scipy.io

TEST_SIZE = 0.3

class TrainDataset(Data.Dataset):
    def __init__(self, ikea_dir, test=False, loaded=False):
        super(TrainDataset, self).__init__()
        self.CWD_PATH = ikea_dir
        self.count = 0
        self.total_voxels = []
        self.total_images = []

        if test:
            name = 'test'
        else:
            name = 'train'

        if not loaded:
            dataSets = DatasetSplit(self.CWD_PATH)
            dataSets.loaddataset(TEST_SIZE)
        print(name+' dataset '+'begin loading==================')
        if not test:
            with open('./DataSet/trainData.json', 'r') as f:
                datadict = json.load(f)
        else:
            with open('./DataSet/testData.json', 'r') as f:   #self.CWD_PATH +
                datadict = json.load(f)

        self.count = datadict['count']
        print(name + ' dataset ' + 'finished loading===============')
        self.total_voxels = datadict['voxels']
        self.total_images = datadict['images']

    def __getitem__(self, index):
        voxel = torch.Tensor(self.total_voxels[index])
        voxel = voxel.squeeze(0)
        image = torch.Tensor(self.total_images[index])
        return image, voxel

    def __len__(self):
        return self.count

class IKEADataset(Data.Dataset):
    def __init__(self, data_dir):
        super(IKEADataset, self).__init__()
        self.data_dir = data_dir
        self.image_list = []
        self.voxel_list = []
        self.image_path = os.path.join(data_dir,'Images')
        self.voxels_path = os.path.join(data_dir,'Voxels')
        self.list_path = os.path.join(data_dir, 'Lists')
        images = open(self.list_path + '/Images.txt')
        voxels = open(self.list_path + '/ModelID.txt')
        for line in images.readlines():
            self.image_list.append(line.strip())
        for line in voxels.readlines():
            self.voxel_list.apend(line.strip()+'.mat')

    def __getitem__(self, index):
        img = sio.imread(os.path.join(self.image_path, self.image_list[index]))
        voxel = scipy.io.loadmat(os.path.join(self.voxels_path, self.voxel_list[index]))
        voxel = torch.Tensor(voxel['grid'])
        img = torch.Tensor(img)
        return img, voxel

    def __len__(self):
        return len(self.image_list)

class ShapeNetDataset(Data.Dataset):
    def __init__(self, data_dir, test=False, zone='all'):
        super(ShapeNetDataset, self).__init__()
        if test:
            self.list_dir = os.path.join(data_dir, 'splits/test')
        else:
            self.list_dir = os.path.join(data_dir, 'splits/train')
        self.data_list = []
        self.data_dir = data_dir

        if zone == 'all':
            files = os.listdir(data_dir)
            for file_name in files:
                file = open(os.path.join(self.list_dir, file_name), 'r')
                for line in file.readlines():
                    self.data.append(line.strip())
        else:
            # dict = {'Chair':'03001627'.'Table':'04379243','Sofa':'04256520', 'Cabinet':'02933112', 'Bed':'02818832'}
            file_name = zone
            file = open(self.list_dir + file_name, 'r')
            for line in file.readlines():
                self.data.append(line.strip())

    def __getitem__(self, index):
        data_name = self.data_list[index]
        voxel =
        voxel = torch.Tensor(self.total_voxels[index])
        voxel = voxel.squeeze(0)
        image = torch.Tensor(self.total_images[index])
        return image, voxel

    def __len__(self):
        return len(self.data)

# class TrainDataset(Data.Dataset):
#     def __init__(self, ikea_dir, loaded, test):
#         super(TrainDataset, self).__init__()
#         files = os.listdir(ikea_dir + '/data_pair/')
#         self.data_files = []
#         for file in files:
#             if not os.path.isdir(ikea_dir + '/data_pair/' + file):
#                 _, extension = os.path.splitext(file)
#                 if extension == '.h5':
#                     self.data_files.append(file)
#         self.pairs = h5py.File(ikea_dir + '/data_pair/batch_1.h5', 'r')
#         self.images = self.pairs['data'].value
#         self.voxels = self.pairs['label-voxel'].value
#
#     def __getitem__(self, index):
#         voxel = torch.Tensor(self.voxels[index])
#         voxel = voxel.squeeze(0)
#         image = torch.Tensor(self.images[index])
#         return image, voxel
#
#     def __len__(self):
#         return self.images.shape[0]