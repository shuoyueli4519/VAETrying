import torch
import random
from model import VAE
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from MyDatasets.celeba_dataset import *
from torch.autograd import Variable

'''
@brief: face_VAE类,其中get_img用于获取图片, get_feature用于获取encoder的输出,
        get_rand_face用于生成随机人脸, get_reconstruction用于重构人脸
'''
class face_VAE():
    def __init__(self):
        self.auto = VAE()
        self.auto.load_state_dict(torch.load("./autoencoder.pkl", map_location='cpu', weights_only=True))
        self.auto.eval()

    def get_img(self, img_path):
        im = Image.open(img_path)
        im = np.array(im)
        im = np.array(Image.fromarray(im).resize((64,64)))
        im = im / 255
        im = im.transpose((2, 0, 1))
        im = torch.from_numpy(im)
        im = im.unsqueeze(0)
        im = im.float()
        return im

    def get_feature(self, img):
        mu, log_var = self.auto.encode(img)
        feature = self.auto.reparameterize(mu, log_var)
        return feature

    def get_rand_face(self, length=128):
        sample = Variable(torch.randn(64, length))  # 随机生成64个样本
        sample = self.auto.decode(sample).cpu()
        rand_name = random.randint(0, 100)
        save_image(sample.data.view(64, 3, 64, 64), './imgs/' + str(rand_name) + '.png')

    def get_reconstruction(self):
        # 需要重构多少图片就设置多少batch_size,即将10换成其它数字
        test_dataset = CelebaDataset(root_dir='./Img/img_align_celeba', resize_dim=64)
        test_loader = DataLoader(test_dataset, batch_size=10, num_workers=1, shuffle=True)
        for i, data in enumerate(test_loader):
            data = data.float()
            feature = self.get_feature(data)
            recon_batch = self.auto.decode(feature).cpu()
            if i == 0:
                comparison = torch.cat([data,
                                        recon_batch.view(10, 3, 64, 64)])
                save_image(comparison.data.cpu(),
                           './imgs/reconstruction' +  '.png', nrow=10)
                break

if __name__ == "__main__":
    if os.path.exists("./imgs") == False:
        os.mkdir("./imgs")

    face = face_VAE()
    face.get_rand_face()
    face.get_reconstruction()


