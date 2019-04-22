import torch


class ParaSetting:
    def __init__(self):
        self.batch_size = 16
        self.label_size = 10
        self.epoch_num = 20
        self.sample_rate = 16000
        self.use_cuda = True
        self.cuda = torch.cuda.is_available() and self.use_cuda
        self.log_step = None
        self.dataset_len = None
        self.dictionary = {0: 'pop',
                           1: 'metal',
                           2: 'disco',
                           3: 'blues',
                           4: 'reggae',
                           5: 'classical',
                           6: 'rock',
                           7: 'hiphop',
                           8: 'country',
                           9: 'jazz'}

        self.TRAIN_DATA_PATH = '../datasets/train.h5'
        self.VAL_DATA_PATH = '../datasets/valid.h5'
        self.TEST_DATA_PATH = '../datasets/test.h5'
        self.MODEL_SAVE_PATH_1 = '../model/best_model_1.pt'
        self.MODEL_SAVE_PATH_2 = '../model/best_model_2.pt'

        if self.cuda:
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            self.kwargs = {}

    def __str__(self):
        out_string = "The Batch Size is {0}\n" \
                     "The Label Size is {1}\n" \
                     "The Epoch Num is {2}\n" \
                     "The Cuda is set to {3}\n" \
                     "The log step is {4}".format(self.batch_size,
                                                  self.label_size,
                                                  self.epoch_num,
                                                  self.cuda,
                                                  self.log_step)
        return out_string


Para = ParaSetting()

if __name__ == '__main__':
    print(Para)
