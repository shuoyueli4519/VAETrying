# VAETrying
## 文件结构
- VAETrying <br/>
|- Img/img_align_celeba(celeba数据集)<br/>
|- imgs(生成和重构结果)<br/>
|- MyDatasets/celeba_dataset.py<br/>
|- autoencoder.pkl(训练好的模型)<br/>
|- model.py<br/>
|- README.md<br/>
|- run.py<br/>
|- train.py<br/>
## 文件详情
- ./MyDatasets中是数据集的dataloader
- model.py是VAE模型构建文件
- train.py是训练代码,更改数据集路径即可训练
- run.py是推理代码,包含重构和生成