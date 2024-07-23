import torch
import torch.nn as nn
from torch.autograd import Variable

'''
@brief: VAE model类,其中mean和var并不是直接的输出,而是通过Linear层进行学习的,
        而且根据是否处于训练状态决定mean和var是否会进行重参数化,每次重参数化会生成10个,
        目的是为了增加样本多样性
@warning: var并不是直接指方差sigma,而是log(sigma^2),这样可以避免sigma为负数
'''

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 128
        hidden_dims = [32, 64, 128, 256, 512]
        modules = []
        in_channels = 3

        # 3*64*64图片经历32*32*32到64*16*16到128*8*8到256*4*4到512*2*2
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU()
                )
            )
            in_channels = dim
        self.encoder = nn.Sequential(*modules)

        # 512*2*2图片全链接到128维的mean和var,代表128对正态分布的叠加
        self.fc_mean = nn.Linear(hidden_dims[-1]*2*2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*2*2, latent_dim)

        modules = []
        self.decoder_back = nn.Linear(latent_dim, hidden_dims[-1]*2*2)
        hidden_dims.reverse()
        hidden_dims = hidden_dims[1:]

        # 512*2*2图片经历256*4*4到128*8*8到64*16*16到32*32*32
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels=dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU()
                )
            )
            in_channels = dim
        self.decoder = nn.Sequential(*modules)

        # 32*32*32图片经历32*64*64最后到3*64*64
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mean = self.fc_mean(result)
        var = self.fc_var(result)
        
        return [mean, var]
    
    def decode(self, latten):
        result = self.decoder_back(latten)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result
    
    def reparameterize(self, mean, var):
        if self.training:
            sample_z = []
            for _ in range(10):
                # 由于var是log(sigma^2),所以var.mul(0.5).exp_()是sigma
                # eps是从标准正态分布中采样的
                # 10是为了增加样本多样性,训练时可以生成多张图片进行反向传播
                std = var.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mean))
            return sample_z

        else:
            return mean
        
    def forward(self, input):
        mean, var = self.encode(input)
        latten = self.reparameterize(mean, var)
        if self.training:
            return [self.decode(i) for i in latten], mean, var
        else:
            return self.decode(latten), mean, var