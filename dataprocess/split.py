import os
import numpy as np
import h5py
import json

class DatasetSplit():
    def __init__(self, ikea_dir):
        self.CWD_PATH = ikea_dir
        self.train_count = 0
        self.test_count = 0
        self.test_images = []
        self.test_voxels = []
        self.train_images = []
        self.train_voxels = []
        self.data_files = []

    def loaddataset(self, test_size=0.3):
        files = os.listdir(self.CWD_PATH + '/data_pair/')
        for file in files:
            if not os.path.isdir(self.CWD_PATH + '/data_pair/' + file):
                _, extension = os.path.splitext(file)
                if extension == '.h5':
                    self.data_files.append(file)
        for file in self.data_files:
            pairs = h5py.File(self.CWD_PATH + '/data_pair/' + file, 'r')
            images = pairs['data']
            voxels = pairs['label-voxel']
            self.test_count = self.test_count + int(images.shape[0] * test_size)
            self.train_count = self.train_count + images.shape[0] - int(images.shape[0] * test_size)
            self.trainTestSplit(images, voxels, test_size)
            # print('+++++++++++')

        dictobj_test = {
                    'images': np.array(self.test_images).tolist(),
                    'voxels': np.array(self.test_voxels).tolist(),
                    'count': self.test_count
        }

        dictobj_train = {
                    'images': np.array(self.train_images).tolist(),
                    'voxels': np.array(self.train_voxels).tolist(),
                    'count': self.train_count
        }

        jsObjTest = json.dumps(dictobj_test)
        jsObjTrain = json.dumps(dictobj_train)

        fileObject = open('/dev/shm/GPVoxelsModel//DataSet/trainData.json', 'w')
        fileObject.write(jsObjTrain)
        fileObject.close()
        print('Train dataset loaded.')
        fileObject = open('/dev/shm/GPVoxelsModel//DataSet/testData.json', 'w')
        fileObject.write(jsObjTest)
        fileObject.close()
        print('Test dataset loaded.')

    def trainTestSplit(self, image,voxel,test_size=0.3):
        images = image
        voxels = voxel
        X_num = images.shape[0]
        train_index = range(X_num)
        train_index = list(train_index)
        test_index = []
        test_num = int(X_num*test_size)
        for i in range(test_num):
            randomIndex = int(np.random.uniform(0, len(train_index)))
            test_index.append(train_index[randomIndex])
            del train_index[randomIndex]

        for item in test_index:
            self.test_images.append(images[item])
            self.test_voxels.append(voxels[item])
        for item in train_index:
            self.train_images.append(images[item])
            self.train_voxels.append(voxels[item])

