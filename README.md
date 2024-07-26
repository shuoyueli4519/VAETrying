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
## DDP单卡多GPU训练命令
- python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 train.py(需要多少张GPU就将nproc_per_node改为多少并在ddp_train.py中修改GPU编号)
- 如果出现**Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.**warning，是由于默认每个进程使用一个线程从而防止过多的线程可能导致上下文切换频繁，增加 CPU 负担，可以使用**export OMP_NUM_THREADS=2**来增加，但有一定风险
- 如果出现**Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())**warning，是由于检查模型中是否存在未使用的模型参数，可以将find_unused_parameters=True设置为False，但其默认值是True，最好不改动