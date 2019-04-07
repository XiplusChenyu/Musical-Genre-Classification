import torch


class Para:
    batch_size = 8
    label_size = 10
    epoch_num = 20
    use_cuda = True
    cuda = torch.cuda.is_available() and use_cuda
    TRAIN_DATA_PATH = '../datasets/train.h5'

    if cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}
