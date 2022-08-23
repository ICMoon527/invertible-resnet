from audioop import minmax
from fileinput import filename
from posixpath import dirname
from typing import Any, Tuple
from typing_extensions import dataclass_transform
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import torch

DIFFUSION_MIN_MAX = (0.0, 0.16199602072082317)
DIFFUSION_MEAN_STD = (0.007952842710335403, 0.049901370936519106)

class DiffusionDataset(Dataset):

    def __init__(self,
                root: str,
                dir_name: list=[],
                train: bool=True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                extract_shape = (32, 32),
                density_estimation = True
                ) -> None:
        super(DiffusionDataset, self).__init__()

        self.root = root
        self.train = train
        self.data = []
        self.targets = []
        self.extract_shape = extract_shape
        self.transform = transform
        self.target_transform = target_transform

        # Read Geo File
        self.geo_data = np.load(root + 'h.npy')
        self.geo_data = self.extractMatrix2D(self.geo_data, self.extract_shape)
        self.geo_data = np.expand_dims(self.geo_data, 0).repeat(300, axis=0)  # (300, 32, 32)
        
        if train:
            assert len(dir_name)!=0, 'No File List.'
            for dir in dir_name:
                npy_files = os.listdir(root + dir)

                for npy_file in npy_files:  # 0_180.npy
                    print('Now processing {}.npy'.format(npy_file))

                    my_data = np.fromfile(root+dir+'/'+npy_file, dtype='float').reshape((300,568,693))
                    my_data = self.extractMatrix3D(my_data, self.extract_shape)  # (300,568,693) -> (300, 32, 32) for example
                    
                    wind_direction = int(npy_file.split('.')[0][2:])  # degree
                    wind_direction = wind_direction / 180 * math.pi  # to radius
                    wind_direction = math.cos(wind_direction)
                    wind_direction_matrix = np.ones_like(my_data) * wind_direction  # (300, 32, 32)

                    my_data = np.stack([my_data, wind_direction_matrix, self.geo_data], axis=1)  # stack other feature to raw data (batch, channel, height , width)
                    
                    # for Forward and Reconstrustion
                    if density_estimation:
                        my_target = my_data[1:, :, :, :]
                        my_data = my_data[0:-1, :, :, :]
                        self.targets.append(my_target)

                    self.data.append(my_data)
                    
                    # # for classification
                    # data_class = int(npy_file.split('_')[0])
                    # self.targets.extend((np.ones(shape=my_data.shape[0], dtype=np.int)*data_class).tolist())

            self.data = np.vstack(self.data)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.targets = np.vstack(self.targets)
            self.targets = self.targets.transpose((0, 2, 3, 1))

            # min max scale for data and target
            temp = self.data[:, :, :, 0]
            self.data[:, :, :, 0] = (temp-np.min(temp)) / (np.max(temp)-np.min(temp))
            temp = self.targets[:, :, :, 0]
            self.targets[:, :, :, 0] = (temp-np.min(temp)) / (np.max(temp)-np.min(temp))

            print('source_data_shape: ', self.data.shape)
            print('target_shape: ', self.targets.shape)
            self.MinMax(self.data[:, :, :, 0])
            self.MeanStd(self.data[:, :, :, 0])
                    
                    
        else:  # test
            assert len(dir_name)!=0, 'No File List.'
            for dir in dir_name:
                npy_files = os.listdir(root + dir)

                for npy_file in npy_files:  # 0_180.npy
                    print('Now processing {}.npy'.format(npy_file))

                    my_data = np.fromfile(root+dir+'/'+npy_file, dtype='float').reshape((300,568,693))
                    my_data = self.extractMatrix3D(my_data, self.extract_shape)  # (300,568,693) -> (300, 32, 32) for example
                    
                    wind_direction = int(npy_file.split('.')[0][2:])  # degree
                    wind_direction = wind_direction / 180 * math.pi  # to radius
                    wind_direction = math.cos(wind_direction)
                    wind_direction_matrix = np.ones_like(my_data) * wind_direction  # (300, 32, 32)

                    my_data = np.stack([my_data, wind_direction_matrix, self.geo_data], axis=1)  # stack other feature to raw data (batch, channel, height , width)
                    
                    # for Forward and Reconstrustion
                    if density_estimation:
                        my_target = my_data[1:, :, :, :]
                        my_data = my_data[0:-1, :, :, :]
                        self.targets.append(my_target)

                    self.data.append(my_data)
                    
                    # # for classification
                    # data_class = int(npy_file.split('_')[0])
                    # self.targets.extend((np.ones(shape=my_data.shape[0], dtype=np.int)*data_class).tolist())

            self.data = np.vstack(self.data)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            self.targets = np.vstack(self.targets)
            self.targets = self.targets.transpose((0, 2, 3, 1))

            # min max scale for data and target
            temp = self.data[:, :, :, 0]
            self.data[:, :, :, 0] = (temp-np.min(temp)) / (np.max(temp)-np.min(temp))
            temp = self.targets[:, :, :, 0]
            self.targets[:, :, :, 0] = (temp-np.min(temp)) / (np.max(temp)-np.min(temp))

            print('source_data_shape: ', self.data.shape)
            print('target_shape: ', self.targets.shape)
            self.MinMax(self.data[:, :, :, 0])
            self.MeanStd(self.data[:, :, :, 0])

            # self.__getitem__(0)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        data, target = self.data[index], self.targets[index]  # (32, 32, 3)
        # data, target = Image.fromarray(data), Image.fromarray(target)

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # RuntimeError: expected scalar type Double but found Float  √
        data, target = data.float(), target.float()  # float64 to float32
        # print(data.shape)
        
        return data, target  # (3, 32, 32) by torch, but (32, 32, 3) by numpy

    def __len__(self) -> int:
        return len(self.data)

    def MinMax(self, array):
        print('MIN: {}, MAX: {}'.format(np.min(array), np.max(array)))
        return np.min(array), np.max(array)

    
    def MeanStd(self, array):
        print('MEAN: {}, STD: {}'.format(np.mean(array), np.std(array)))
        return np.mean(array), np.std(array)


    def extractMatrix2D(self, array, shape):
        new_array = np.zeros(shape)
        delta_x = array.shape[0] // shape[0]
        delta_y = array.shape[1] // shape[1]

        for i in range(shape[0]):
            for j in range(shape[1]):
                new_array[i, j] = array[delta_x*i, delta_y*j]

        return new_array

    def extractMatrix3D(self, array, shape):
        new_array = np.zeros((array.shape[0], shape[0], shape[1]))
        
        for i in range(new_array.shape[0]):
            new_array[i] = self.extractMatrix2D(array=array[i], shape=shape)

        return new_array


if __name__ == '__main__':
    # dataset = DiffusionDataset('/data/langjunwei/taizhou/data/', ['0/'], True)
    dataset = DiffusionDataset('/data/langjunwei/taizhou/data/', ['0/', '1/', '2/', '3/', '4/'], True)
    dataset.__getitem__(0)

    # from torchvision.datasets.cifar import CIFAR10
    # dataset = CIFAR10(root='./data', train=True, download=True)
    # item = dataset.__getitem__(3)
    # print(item)