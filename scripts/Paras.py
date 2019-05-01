import torch


class ParaSetting:
    def __init__(self):
        self.batch_size = None
        self.label_size = 10
        self.epoch_num = None
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
        self.r_dictionary = {'pop': 0,
                             'metal': 1,
                             'disco': 2,
                             'blues': 3,
                             'reggae': 4,
                             'classical': 5,
                             'rock': 6,
                             'hiphop': 7,
                             'country': 8,
                             'jazz': 9}

        # On pure GTZAN dataset
        self.TRAIN_DATA_PATH = '../datasets/train.h5'
        self.VAL_DATA_PATH = '../datasets/valid.h5'
        self.TEST_DATA_PATH = '../datasets/test.h5'

        # On hybrid dataset
        self.A_TRAIN_DATA_PATH = '../datasets/fin_train.h5'
        self.A_VAL_DATA_PATH = '../datasets/fin_valid.h5'
        self.A_TEST_DATA_PATH = '../datasets/fin_test.h5'

        # On hybrid dataset
        self.LA_TRAIN_DATA_PATH = '../datasets/l_train.h5'
        self.LA_VAL_DATA_PATH = '../datasets/l_valid.h5'
        self.LA_TEST_DATA_PATH = '../datasets/l_test.h5'

        self.MODEL_SAVE_FOlD = '../model/'
        self.LOG_SAVE_FOLD = '../log/'

        if self.cuda:
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            self.kwargs = {}

        self.learning_rate = 1e-5

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
