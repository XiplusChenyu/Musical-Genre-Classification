import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from Paras import Para


class TorchData(Dataset):

    def __init__(self, dataset_path):
        """
        Take the h5py dataset
        """
        super(TorchData, self).__init__()
        self.dataset = h5py.File(dataset_path, 'r')
        self.mel = self.dataset['mel']
        self.tag = self.dataset['tag']

        self.len = self.mel.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mel = torch.from_numpy(self.mel[index].astype(np.float32))
        tag = torch.from_numpy(self.tag[index].astype(np.float32))

        # reshape the label
        mel = mel.transpose(0, 1)
        sample = {"mel": mel, "tag": tag}
        return sample


# define the data loaders
def torch_dataset_loader(dataset, batch_size, shuffle, kwargs):
    """
    take the h5py dataset
    """
    loader = DataLoader(TorchData(dataset),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        **kwargs)
    return loader


train_loader = torch_dataset_loader(Para.TRAIN_DATA_PATH, Para.batch_size, True, Para.kwargs)

if __name__ == '__main__':

    for index, data_item in enumerate(train_loader):
        print(data_item['mel'].shape)
        print(data_item['tag'].shape)
