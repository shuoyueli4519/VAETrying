import torch
import torch.nn.functional as F
import argparse
from MyDatasets.celeba_dataset import *
from model import VAE
from torch.autograd import Variable
import os
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

'''
@brief: 自定义的损失函数,重构误差需求重构的图片与原图片的差距尽可能小,使用BCE可以实现这一点,
        损失函数相当于BCE和KLD的加权和,即BCE + lambda * KLD
@DDP:   使用DDP进行单卡多GPU分布式训练时,需要指定nccl作为后端,并且使用torch.distributed.init_process_group(backend='nccl')初始化进程组,
        使用torch.distributed.get_rank()获取当前进程的rank,使用torch.cuda.set_device(opt.local_rank)设置当前进程的GPU,
        使用torch.nn.parallel.DistributedDataParallel()包装模型,使用sampler=DistributedSampler()包装数据集
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
    parser.add_argument('--epoch', type=int, default=5, help='epoch size')
    parser.add_argument('--local_rank', type=int, default=-1)
    opt = parser.parse_args()

    torch.distributed.init_process_group(backend='nccl')
    opt.local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(opt.local_rank)
    device = torch.device("cuda", opt.local_rank)

    LR = opt.lr
    BATCH_SIZE = opt.batch_size
    EPOCHES = opt.epoch
    LOG_INTERVAL = 5    # 每5个batch打印一次训练信息
    auto = VAE().to(device)
    print(f"ready to train")
    optimizer = torch.optim.Adam(auto.parameters(), lr=LR)
    train_dataset = CelebaDataset(root_dir='./Img/new', resize_dim=64)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    model = torch.nn.parallel.DistributedDataParallel(auto, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)

    for i in range(EPOCHES):

        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.float().to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0 and dist.get_rank() == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item() / len(data)))
        if dist.get_rank() == 0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(i, train_loss / len(train_loader.dataset) * dist.get_world_size()))

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "autoencoder.pkl")