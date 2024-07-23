# 选股模型
### 基于FinBERT模型训练
### 目前能够根据所输入的消息，返回影响的板块，利好有哪些板块，利空有哪些板块，以及重要程度

### 部署环境 windows 安装anaconda 以及 CUDA
https://blog.csdn.net/YYDS_WV/article/details/137825313

https://developer.nvidia.com/rdp/cudnn-archive

1、打开 Anaconda Prompt，创建一个新的虚拟环境
conda create --name pytorch-env python=3.9

2、 激活虚拟环境
conda activate pytorch-env

4、 训练模型并发布为服务  
python CustomDataset.py

### 训练模型
1、 激活 Anaconda 虚拟环境
conda activate pytorch-env
2、 运行 scripts.train_model
python -m scripts.train_model

### 测试模型
1、 激活 Anaconda 虚拟环境
conda activate pytorch-env
2、 运行 scripts.train_model
python -m scripts.test_model

### 在线预测服务
1、 激活 Anaconda 虚拟环境
conda activate pytorch-env
2、 运行 app.main
python -m app.main