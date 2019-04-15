import time

from Paras import Para
from model import M_model as M
from data_loader import train_loader


def train():
    start_time = time.time()
    model = M.eval()
    train_loss = 0.
    for batch_idx, data in enumerate(train_loader):
        mel, tag = data['mel'], data['tag']
        print(model(mel))
        break


if __name__ == '__main__':
    train()

