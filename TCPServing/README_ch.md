[English](README.md) | 简体中文
# TCP服务

## 待办
- [x] TCP通信
  - [x] TCP阻塞式通信
  - [ ] TCP非阻塞式通信
- [x] 图片传输
  - [x] 发送图片地址
  - [ ] 发送图片数据

## 使用说明   

### 配置:
在[配置文件](jg_config.py)中设置服务器、推理引擎、检测模型和识别模型的参数。

### 启动:

#### 激活conda环境
```bash   
  conda activate SBCR
```

#### 启动服务器:
```bash
  CUDA_VISIBLE_DEVICES=0 python jg_predict_system.py
 ```
如果启动成功，则显示
```bash
  2020-12-14 10:59:55,199-INFO: Finished initlizing recognition system.
  2020-12-14 10:59:55,214-INFO: Started server system.
  2020-12-14 10:59:55,226-INFO: Host:127.0.0.1, port:1024, recieve buf size:83886080
  2020-12-14 10:59:55,228-INFO: Wait for connection.
```

#### 启动客户端(仅用于测试):
```bash
  python jg_test_client.py
```
