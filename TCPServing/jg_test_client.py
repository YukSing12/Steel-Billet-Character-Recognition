import socket
import sys
from jg_protocol import ClientProtocol
import os
cp = ClientProtocol()
# 创建 socket 对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# 获取本地主机名
host = '127.0.0.1'
buf_szie = 10*1024*1024*8
# 设置端口号
port = 1024

# 连接服务，指定主机和端口
s.connect((host, port))

# 接收小于 10*1024*1024*8 字节的数据
msg = s.recv(10*1024*1024*8)
client_decode_msg = cp.decode(msg)
print(client_decode_msg)

fnc = 0x32
image_file = os.path.join('test.JPG')
image_path = os.path.abspath(image_file)
for i in range(1):
    client_encode_msg = cp.encode(fnc,image_file[-3:],1024,image_path)
    s.send(client_encode_msg)
    msg = s.recv(10*1024*1024*8)
    client_decode_msg = cp.decode(msg)
    print(client_decode_msg)
s.shutdown(2)
