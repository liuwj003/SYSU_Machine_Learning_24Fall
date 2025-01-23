import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    define structure of VAE
    """
    def __init__(self, image_channels=3, latent_dim=20, nn_type='Conv'):
        """
        Encoder use CNN structure, Decoder deconvolution
        :param image_channels: MNIST: 1; CIFAR-10: 3
        :param latent_dim: latent variable z dimension
        :param nn_type: 'Linear' or 'Conv'
        """
        super(VAE, self).__init__()
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.nn_type = nn_type

        # encoder
        if nn_type == 'Linear':
            # several fully-connected linear layers
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(image_channels * 32 * 32, 256),
                # nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                # 直接输出 mean 和 log_var
                nn.Linear(64, latent_dim * 2)
            )

        elif nn_type == 'Conv':
            # Convolution
            self.encoder = nn.Sequential(
                # 搜索得知一般使用较小的 kernel， kernel_size = 3 or 5
                # 针对 32*32的 CIFAR-10 输入进行计算, 并且其实是[batch_size,channels,height,width]格式
                # 3*32*32 -> 32*16*16
                nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # 32*16*16 -> 64*8*8
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # 64*8*8 -> 128*4*4
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # 128*4*4 -> 256*2*2
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            )

        # decoder
        if nn_type == 'Linear':
            # linear
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                # nn.Linear(256, 784),
                nn.Linear(256, image_channels * 32 * 32),
                nn.Sigmoid()
            )

        elif nn_type == 'Conv':
            # Convolution
            self.decoder = nn.Sequential(
                # 256*2*2 -> 128*4*4
                nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
                nn.LeakyReLU(0.2),
                # 128*4*4 -> 64*8*8
                nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
                nn.LeakyReLU(0.2),
                # 64*8*8 -> 32*16*16
                nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                nn.LeakyReLU(0.2),
                # 32*16*16 -> 3*32*32
                nn.ConvTranspose2d(32, image_channels, 3, 2, 1, 1),
                nn.Sigmoid()
            )

        # should flatten first
        self.flatten = nn.Flatten()  # [batch_size,256,2,2] -> [batch_size, 256*2*2]
        # train mean (for CNN output)
        self.fc_mean = nn.Linear(in_features=256*2*2, out_features=latent_dim)
        # train log_variance (for CNN output)
        self.fc_log_var = nn.Linear(in_features=256*2*2, out_features=latent_dim)
        # 要先把隐变量恢复成 256*2*2的格式 (for CNN output)
        # [batch_size, latent_dim] -> [batch_size, 256*2*2]
        self.recover = nn.Linear(in_features=self.latent_dim, out_features=256*2*2)


    def reparameterize(self, mean, log_var):
        """
        reparameterization, to get z = mean + var * noise
        :param mean:
        :param log_var:
        :return: latent variable z
        """
        sqrt_var = torch.exp(0.5 * log_var)
        noise = torch.randn_like(sqrt_var)
        z = mean + sqrt_var * noise
        return z

    def forward(self, x):
        """
        forward: x -> z -> recon_x
        :param x: sample x
        :return: recon_x, latent_mean, latent_log_var
        """
        if self.nn_type == 'Conv':
            # encode
            e = self.encoder(x)  # [batch_size, 256, 2, 2]
            e = self.flatten(e)  # flatten: [batch_size, 256*2*2]
            mean = self.fc_mean(e)  # [batch_size, latent_dim]
            log_var = self.fc_log_var(e)  # [batch_size, latent_dim]

            # latent variable
            z = self.reparameterize(mean, log_var)

            # decode
            d = self.recover(z)  # [batch_size, latent_dim] -> [batch_size, 256*2*2]
            # 要重塑为 4 维张量：[batch_size, 256*2*2] -> [batch_size, 256, 2, 2]
            d = d.view(-1, 256, 2, 2)
            recon_x = self.decoder(d)

        elif self.nn_type == 'Linear':
            # encode
            e = self.encoder(x)
            mean, log_var = e[:, :self.latent_dim], e[:, self.latent_dim: ]

            # latent variable
            z = self.reparameterize(mean, log_var)

            # decode
            flatten_x = self.decoder(z)
            recon_x = flatten_x.view(x.size(0), self.image_channels, 32, 32)

        return recon_x, mean, log_var

    def generate(self, z):
        """
        generate new x from latent variable z
        :param z: latent variable with noise
        :return: generated x
        """
        if self.nn_type == 'Conv':
            # decode
            d = self.recover(z)  # [batch_size, latent_dim] -> [batch_size, 256*2*2]
            d = d.view(-1, 256, 2, 2)  # [batch_size, 256*2*2] -> [batch_size, 256, 2, 2]
            generated_x = self.decoder(d)

        elif self.nn_type == 'Linear':
            # decode
            flatten_x = self.decoder(z)
            generated_x = flatten_x.view(-1, self.image_channels, 32, 32)

        return generated_x





