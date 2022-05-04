# https://pytorch.org/vision/stable/datasets.html
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os 
import h5py 
import numpy as np
from tqdm import tqdm

import torch 
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

PATH = '/data1/jcampos/HAWQ-main/data/train'

FEATURES = [
    b'j_pt', b'j_eta', b'j_mass', b'j_tau1_b1',
    b'j_tau2_b1', b'j_tau3_b1', b'j_tau1_b2', b'j_tau2_b2',
    b'j_tau3_b2', b'j_tau32_b1', b'j_tau32_b2', b'j_zlogz',
    b'j_c1_b0', b'j_c1_b1', b'j_c1_b2', b'j_c2_b1'
]

LABELS = [
    'j_g', 'j_q', 'j_w', 'j_z', 'j_t'
]


class JetTaggingDataset(Dataset):
    '''Jet Tagging dataset'''
    # TODO
    # apply transforms
    def __init__(self, path, features=FEATURES, transform=None) -> None:
        '''
        Args:
            path (str): Path to dataset.
            features (list[str], optional): Optional only load selected features. 
            transform (callable, optional): Optional transform to be applied on dataset. 
        '''
        self.path = path
        self.features = features
        self.transform = transform

        if os.path.isdir(path):
            self.data, self.labels = self.load_data()
        else:
            raise RuntimeError(f'Path is not a directory: {path}')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _load_data(self, files):
        data = np.empty([1, 54])
        labels = np.empty([1, 5])
        files_parsed = 0
        progress_bar = tqdm(files)

        for file in progress_bar:
            file = os.path.join(self.path, file)
            try:
                h5_file = h5py.File(file, 'r')

                if files_parsed == 0:
                    feature_names = np.array(h5_file['jetFeatureNames'])
                    feature_indices = [int(np.where(feature_names == feature)[0]) for feature in self.features]
                
                h5_dataset = h5_file['jets']
                # convert to ndarray and concatenate with dataset
                h5_dataset = np.array(h5_dataset, dtype=np.float)
                # separate data from labels
                np_data = h5_dataset[:, :54]
                np_labels = h5_dataset[:,-5:]
                # update data and labels
                data = np.concatenate((data, np_data), axis=0, dtype=np.float)
                labels = np.concatenate((labels, np_labels), axis=0, dtype=np.float)
                h5_file.close()
                # update progress bar 
                files_parsed += 1
                progress_bar.set_postfix({'files loaded': files_parsed})
            except:
                print(f'Could not load file: {file}')
        
        data = data[:, feature_indices]
        return data[1:].astype(np.float32), labels[1:].astype(np.float32)
    
    def load_data(self):
        files = os.listdir(self.path)
        files = [file for file in files if file.endswith('.h5')]
        if len(files) == 0:
            print('Directory does not contain any .h5 files')
            return None
        return self._load_data(files)


if __name__ == '__main__':
    dataset = JetTaggingDataset(PATH)
    print(dataset[1])
    print(len(dataset))
