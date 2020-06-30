import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as func
import random

from core.Models import Generator, Discriminator


class ModelTrainer:
    def __init__(self, data_dir, im_size, batch_size, disc_save_path,
                 gen_save_path, betas = (0.5, 0.999), use_cuda=True):

        self.device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.gen_save_path = gen_save_path
        self.disc_save_path = disc_save_path

        self.GeneratorModel = Generator().to(self.device)
        self.DiscriminatorModel = Discriminator().to(self.device)

        self.dataset = dset.ImageFolder(root=data_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(im_size),
                                            transforms.CenterCrop(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.G_optim = optim.Adam(self.GeneratorModel.parameters(), lr=0.0002, betas=betas)
        self.D_optim = optim.Adam(self.DiscriminatorModel.parameters(), lr=0.0002, betas=betas)
        self.error = nn.BCELoss()

    def train(self, epochs, save_every):
        self.GeneratorModel.apply(self.weights_init)
        self.DiscriminatorModel.apply(self.weights_init)

        for epoch in range(epochs):
            for i, data in enumerate(self.dataloader, 0):
                self.DiscriminatorModel.zero_grad()
                real = data[0].to(self.device)
                label = torch.full((real.size(0),), 1, device=self.device)
                out = self.DiscriminatorModel(real).view(-1)

                disc_error_real = self.error(out, label)
                disc_error_real.backward()

                res = int(64 * random.uniform(0.9, 1.1))
                input = func.interpolate(func.interpolate(data[0], (res, res)), (128, 128)).to(self.device)
                recovered = self.GeneratorModel(input)

                label.fill_(0)
                out = self.DiscriminatorModel(recovered.detach()).view(-1)

                disc_error_recovered = self.error(out, label)
                disc_error_recovered.backward()

                self.D_optim.step()

                self.GeneratorModel.zero_grad()
                label.fill_(1)
                out = self.DiscriminatorModel(recovered).view(-1)
                gen_error = self.error(out, label)
                gen_error.backward()

                self.G_optim.step()

            if epoch % save_every == 0 and epoch > 0:
                print(epoch)
                self.save_model()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def save_model(self):
        torch.save(self.GeneratorModel, self.gen_save_path + "\\generator")
        torch.save(self.DiscriminatorModel, self.disc_save_path + "\\discriminator")

