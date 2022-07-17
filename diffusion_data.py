from fileinput import filename
from posixpath import dirname
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

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
                    my_data = np.fromfile(root+dir+'/'+npy_file, dtype='float').reshape((300,568*693))
        else:  # test
            pass

if __name__ == '__main__':
    dataset = DiffusionDataset('/data/langjunwei/taizhou/data/', ['0/'])
    # from torchvision.datasets.cifar import CIFAR10
    # dataset = CIFAR10(root='./data', train=True, download=True)
    # item = dataset.__getitem__(3)
    # print(item)