from Paras import Para
import numpy as np
from torch import nn


def accuracy_function(output, target):
    # shape: (batch, labels)
    f_output = output.cpu() if Para.cuda else output.clone()
    f_target = target.cpu() if Para.cuda else target.clone()

    output_res = f_output.detach().numpy()
    target_res = f_target.detach().numpy()
    predicted_index = np.argmax(output_res, axis=1)

    target_index = np.argmax(target_res, axis=1)

    # counter
    correct = np.sum(predicted_index == target_index)
    accuracy = correct / (output.shape[0])
    return accuracy


def matrix_tuple(output, target):
    f_output = output.cpu() if Para.cuda else output.clone()
    f_target = target.cpu() if Para.cuda else target.clone()

    output_res = f_output.detach().numpy()
    target_res = f_target.detach().numpy()
    predicted_index = np.argmax(output_res, axis=1)
    target_index = np.argmax(target_res, axis=1)
    result_list = [[int(predicted_index[i]), int(target_index[i])] for i in range(len(predicted_index))]
    return result_list


def bce_loss(output, target):
    loss_mlp = nn.BCELoss()
    loss = loss_mlp(output, target)
    return loss


if __name__ == '__main__':
    from data_loader import torch_dataset_loader
    train_loader = torch_dataset_loader(Para.TRAIN_DATA_PATH, Para.batch_size, True, Para.kwargs)

    for index, data_item in enumerate(train_loader):
        tag = data_item['tag']
        print(bce_loss(tag, tag), accuracy_function(tag, tag))
        break

