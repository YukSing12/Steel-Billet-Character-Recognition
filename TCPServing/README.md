English | [简体中文](README_ch.md)
# TCPServing

## Todo
- [x] TCP Comunication
  - [x] TCP Blocking Comunication
  - [ ] TCP Nonblocking Comunication
- [x] Image Transmission
  - [x] Send Image Path
  - [ ] Send Image Data

## Instruction   

### Configuration:
Set server, inference engine, detection model, and recognition model in [configuration file](jg_config.py).

### Start:

#### Activate conda environment
```bash   
  conda activate SBCR
```

#### Start server:
```bash
  CUDA_VISIBLE_DEVICES=0 python jg_predict_system.py
 ```
If started successfully, it will show that
```bash
  2020-12-14 10:59:55,199-INFO: Finished initlizing recognition system.
  2020-12-14 10:59:55,214-INFO: Started server system.
  2020-12-14 10:59:55,226-INFO: Host:127.0.0.1, port:1024, recieve buf size:83886080
  2020-12-14 10:59:55,228-INFO: Wait for connection.
```
#### Start client (for test):
```bash
  python jg_test_client.py
```
