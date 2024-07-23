import torch
import torch.nn.functional as F
import argparse
from MyDatasets.celeba_dataset import *
from model import VAE
from torch.autograd import Variable

'''
@brief: 自定义的损失函数,重构误差需求重构的图片与原图片的差距尽可能小,使用BCE可以实现这一点,
        损失函数相当于BCE和KLD的加权和,即BCE + lambda * KLD
'''
def loss_function(recon_x, x, mu, logvar):
    BCE = 0
    for recon_x_one in recon_x:
        BCE += F.binary_cross_entropy(recon_x_one.view(-1, 3 * 64 * 64), x.view(-1, 3 * 64 * 64))
    BCE /= len(recon_x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * 3 * 64 * 64

    return BCE + KLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005, help='lr')
    parser.add_argument('--batch_size', type=int, default=144, help='batch size')
    parser.add_argument('--epoch', type=int, default=1, help='epoch size')
    opt = parser.parse_args()

    LR = opt.lr
    BATCH_SIZE = opt.batch_size
    EPOCHES = opt.epoch
    LOG_INTERVAL = 5    # 每5个batch打印一次训练信息
    auto = VAE()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        auto.cuda()

    optimizer = torch.optim.Adam(auto.parameters(), lr=LR)
    train_dataset = CelebaDataset(root_dir='./Img/img_align_celeba', resize_dim=64)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True)
    for i in range(EPOCHES):
        auto.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.float()
            if cuda_available:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = auto(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(i, train_loss / len(train_loader.dataset)))

    torch.save(auto, "autoencoder.pkl")