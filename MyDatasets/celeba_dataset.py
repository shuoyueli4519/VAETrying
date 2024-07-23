import os
import torchvision

import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

'''
@brief: 数据集使用celeba数据集,其中__init__中的image_files[:20000]是为了减少数据集大小,
        选取前20000张图片,可以根据自己的需求更改
'''
class CelebaDataset(Dataset):
    def __init__(self, root_dir, resize_dim):
        self.root_dir = root_dir
        self.resize_dim = resize_dim

        image_files = os.listdir(self.root_dir)
        self.image_name_list = image_files[:20000]

        transform = transforms.Compose([transforms.Resize((self.resize_dim, self.resize_dim)),
                                        transforms.ToTensor()])
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.image_name_list[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.image_name_list)
    
if __name__ == '__main__':
    celeba_dataset = CelebaDataset(root_dir='./Img/img_align_celeba', resize_dim=64)
    print(celeba_dataset[1].shape)
    