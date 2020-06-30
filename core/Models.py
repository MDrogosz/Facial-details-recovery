import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.cnn1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(3)
        self.relu1 = nn.PReLU()

        self.cnn2 = nn.Conv2d(in_channels=3, out_channels=2 * 8 ** 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(2 * 8 ** 2)
        self.relu2 = nn.PReLU()

        self.cnn3 = nn.Conv2d(in_channels=2 * 8 ** 2, out_channels=3 * 8 ** 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN3 = nn.BatchNorm2d(3 * 8 ** 2)
        self.relu3 = nn.PReLU()

        self.cnn4 = nn.Conv2d(in_channels=3 * 8 ** 2, out_channels=8 ** 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN4 = nn.BatchNorm2d(8 ** 3)
        self.relu4 = nn.PReLU()

        self.cnn5 = nn.ConvTranspose2d(in_channels=8 ** 3, out_channels=3 * 8 ** 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.BN5 = nn.BatchNorm2d(3 * 8 ** 2)
        self.relu5 = nn.PReLU()

        self.cnn6 = nn.ConvTranspose2d(in_channels=3 * 8 ** 2, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)

        self.tan = nn.Tanh()

    def forward(self, x):

        out = self.cnn1(x)
        out = self.BN1(out)
        out = self.relu1(out)

        out = self.cnn2(out)
        out = self.BN2(out)
        out2 = self.relu2(out)

        out = self.cnn3(out2)
        out = self.BN3(out)
        out3 = self.relu3(out)

        out = self.cnn4(out3)
        out = self.BN4(out)
        out = self.relu4(out)

        out = self.cnn5(out)
        out = self.BN5(out)
        out = torch.add(self.relu5(out), out3/2)

        out = torch.add(self.cnn6(out), x/2)

        out = self.tan(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=3 * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace = True)

        self.cnn2 = nn.Conv2d(in_channels=3 * 8, out_channels=8 ** 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.BN1 = nn.BatchNorm2d(8 ** 2)

        self.cnn3 = nn.Conv2d(in_channels=8 ** 2, out_channels=8 ** 3, kernel_size=4, stride=2, padding=1,
                              bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.BN2 = nn.BatchNorm2d(8 ** 3)

        self.cnn4 = nn.Conv2d(in_channels=8 ** 3, out_channels=8 ** 3, kernel_size=4, stride=2, padding=1,
                              bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.BN3 = nn.BatchNorm2d(8 ** 3)

        self.cnn5 = nn.Conv2d(in_channels=8 ** 3, out_channels=8 ** 3, kernel_size=4, stride=2, padding=1,
                              bias=False)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.BN4 = nn.BatchNorm2d(8 ** 3)

        self.cnn6 = nn.Conv2d(in_channels=8 ** 3, out_channels=3 * 8 ** 2, kernel_size=4, stride=2, padding=1,
                              bias=False)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.BN5 = nn.BatchNorm2d(3 * 8 ** 2)

        self.cnn7 = nn.Conv2d(in_channels=3 * 8 ** 2, out_channels=1, kernel_size=2, stride=1, padding=0,
                              bias=False)
        self.sigmoid = nn. Sigmoid()

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.BN1(out)

        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.BN2(out)

        out = self.cnn4(out)
        out = self.relu4(out)
        out = self.BN3(out)

        out = self.cnn5(out)
        out = self.relu5(out)
        out = self.BN4(out)

        out = self.cnn6(out)
        out = self.relu6(out)
        out = self.BN5(out)

        out = self.cnn7(out)
        out = out.view(out.size(0), -1)

        out = self.sigmoid(out)

        return out
