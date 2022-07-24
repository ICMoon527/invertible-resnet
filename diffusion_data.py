from fileinput import filename
from posixpath import dirname
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math

class DiffusionDataset(Dataset):

    def __init__(self,
                root: str,
                dir_name: list=[],
                train: bool=True
                ) -> None:
        super(DiffusionDataset, self).__init__()

        self.root = root
        self.train = train
        
        if train:
            assert len(dir_name)!=0, 'No File List.'
            for dir in dir_name:
                npy_files = os.listdir(root + dir)
                for npy_file in npy_files:
                    my_data = np.fromfile(root+dir+'/'+npy_file, dtype='float').reshape((300,568,693))
                    my_data = self.extractMatrix3D(my_data, (32, 32))  # (300,568,693) -> (300, 32, 32) for example
                    
                    wind_direction = int(npy_file.split('.')[0][2:])  # degree
                    wind_direction = wind_direction / 180 * math.pi  # to radius
                    wind_direction = math.cos(wind_direction)
                    
        else:  # test
            pass


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
    dataset = DiffusionDataset('/data/langjunwei/taizhou/data/', ['0/'])
    # from torchvision.datasets.cifar import CIFAR10
    # dataset = CIFAR10(root='./data', train=True, download=True)
    # item = dataset.__getitem__(3)
    # print(item)