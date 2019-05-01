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
        mel = self.mel[index].astype(np.float32)
        mel = np.reshape(mel, (1, mel.shape[0], mel.shape[1]))
        mel = torch.from_numpy(mel)
        tag = torch.from_numpy(self.tag[index].astype(np.float32))
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


if __name__ == '__main__':
    train_loader = torch_dataset_loader(Para.TRAIN_DATA_PATH, Para.batch_size, True, Para.kwargs)
    validation_loader = torch_dataset_loader(Para.VAL_DATA_PATH, Para.batch_size, False, Para.kwargs)
    test_loader = torch_dataset_loader(Para.TEST_DATA_PATH, Para.batch_size, False, Para.kwargs)

    fin_train_loader = torch_dataset_loader(Para.A_TRAIN_DATA_PATH, Para.batch_size, True, Para.kwargs)
    fin_validation_loader = torch_dataset_loader(Para.A_VAL_DATA_PATH, Para.batch_size, False, Para.kwargs)
    fin_test_loader = torch_dataset_loader(Para.A_TEST_DATA_PATH, Para.batch_size, False, Para.kwargs)

    l_train_loader = torch_dataset_loader(Para.LA_TRAIN_DATA_PATH, Para.batch_size, True, Para.kwargs)
    l_validation_loader = torch_dataset_loader(Para.LA_VAL_DATA_PATH, Para.batch_size, False, Para.kwargs)
    l_test_loader = torch_dataset_loader(Para.LA_TEST_DATA_PATH, Para.batch_size, False, Para.kwargs)

    for index, data_item in enumerate(fin_train_loader):
        print(data_item['mel'].shape)
        print(data_item['tag'].shape)
        break
