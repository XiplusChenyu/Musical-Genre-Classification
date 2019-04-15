import torch


class ParaSetting:
    def __init__(self):
        self.batch_size = 8
        self.label_size = 10
        self.epoch_num = 20
        self.use_cuda = True
        self.cuda = torch.cuda.is_available() and self.use_cuda
        self.TRAIN_DATA_PATH = '../datasets/train.h5'
        if self.cuda:
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            self.kwargs = {}

    def __str__(self):
        out_string = "The Batch Size is {0}\n" \
                     "The Label Size is {1}\n" \
                     "The Epoch Num is {2}\n" \
                     "The Cuda is set to {3}\n".format(self.batch_size, self.label_size, self.epoch_num, self.cuda)
        return out_string


Para = ParaSetting()

if __name__ == '__main__':
    print(Para)
