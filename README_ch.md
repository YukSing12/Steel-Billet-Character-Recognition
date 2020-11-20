[English](README.md) | 简体中文
# 钢胚字符识别，SBCR

## 介绍
钢胚字符识别

## 整体目录结构
Steel-Billet-Character-Recognition 的整体目录结构介绍如下：

Steel-Billet-Character-Recognition   
├── [Xiang-Steel-Billet-Dataset](https://github.com/YukSing12/Xiang-Steel-Billet-Dataset)    (尚未开源)   
├── [Tangshan-Steel-Billet-Dataset](https://github.com/YukSing12/Tangshan-Steel-Billet-Dataset)    (尚未开源)   
├── [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)    
├── [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)    
├── predict.py   
└── results  

## 待办    

- [x] 模型选择   
  - [x] 检测：DB([paper](https://arxiv.org/abs/1911.08947))
  - [x] 识别：CRNN([paper](https://arxiv.org/abs/1507.05717))
- [x] 模型训练     
- [ ] 模型优化   
  - [ ] 模型压缩    
  - [ ] 模型量化    
- [ ] 模型部署    

## 数据集
### 湘钢数据集   
<div align="center">
    <img src="Xiang-Steel-Billet-Dataset/train_image/BXAIa2019082512471601.jpg" width="400">
</div>

### 唐钢数据集       
<div align="center">
    <img src="output/01440.JPG" width=400>
    <img src="output/01510.JPG" width=400>
</div>
