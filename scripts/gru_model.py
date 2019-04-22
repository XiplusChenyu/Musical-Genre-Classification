import torch.nn as nn
import torch
torch.manual_seed(1)


class GruModel(nn.Module):
    def __init__(self):
        super(GruModel, self).__init__()
        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        self.GruLayer = nn.GRU(input_size=2048,
                               hidden_size=256,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)

        self.GruLayerF = nn.Sequential(nn.BatchNorm1d(2048),
                                       nn.Dropout(0.6))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=2048, out_features=512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # _input (batch_size, time, freq)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        out = self.convBlock3(out)
        # [16, 256, 8, 8]

        out = out.contiguous().view(out.size()[0], out.size()[2], -1)
        out, _ = self.GruLayer(out)
        out = out.contiguous().view(out.size()[0],  -1)
        # out_features=4096

        out = self.GruLayerF(out)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        return out


G_model = GruModel()

if __name__ == '__main__':
    print(G_model)



