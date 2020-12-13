import os
import sys
import numpy as np
import struct
import cv2
from collections import OrderedDict
import codecs
class ServerProtocol(object):
    """
    规定：
        数据包头部占4字节
        报文总长度占2字节
        功能码占1字节
        数据域不定长
        结束符占1字节
        高字节在前，低字节在后
    """
    def __init__(self, bs=None):
        """
        如果bs为None则代表需要创建一个数据包
        否则代表需要解析一个数据包
        """
        if bs:
            self.bs = bytearray(bs)
        else:
            self.bs = bytearray(0)
    
    def encode(self,fnc,results,img_format,img_size,img_path):
        self.bs = bytearray(0)

        head = 'JGIR'
        self.bs += bytearray(head.encode(encoding='utf8'))

        # len: head(4) + ln(2) + fnc(1) + num(1) + num*structure + format(3) + img_len(4) + img_path(n) + end(1)
        ln = (len(head.encode(encoding='utf8')) + 2 + 1 + 1 + len(results)*(2+4+11+4+4+4+4+1) + 3 + 4 + len(img_path.encode(encoding='utf8')) + 1)

        if len(img_format.encode(encoding='utf8')) != 3:
            raise Exception("Invalid image format.")

        self.bs += (bytearray(ln.to_bytes(2, byteorder='little')) +
                    bytearray(fnc.to_bytes(1, byteorder='little')) +
                    bytearray(len(results).to_bytes(1, byteorder='little')))    # 1 Bytes

        for result in results:
            for item in result:
                if type(result[item]) is str:
                    self.bs += bytearray(result[item].encode(encoding='utf8'))
                elif type(result[item]) is float or type(result[item]) is np.float32:
                    self.bs += bytearray(struct.pack("f", result[item]))
                else:
                    raise TypeError()
            self.bs += bytearray(('@').encode(encoding='utf8'))

        self.bs += (bytearray(img_format.encode(encoding='utf8')) +
                   bytearray(img_size.to_bytes(4, byteorder='little')) +
                   bytearray(img_path.encode(encoding='utf8')) +
                   bytearray((0x0a).to_bytes(1, byteorder='little')))
        return self.bs

    def decode(self,bs=None):
        if bs:
            ln = int.from_bytes(bs[4:5], byteorder='little')
            if ln != len(bs):
                print("Warning: ln != len(self.bs)")
                self.bs,_ = codecs.escape_decode(bs,'hex')
            else:
                self.bs = bs
        
        head = self.bs[:4].decode(encoding='utf8')
        if head != 'JGIR':
            return self.bs
        
        end = self.bs[-1]
        if end != 0x0a:
            return self.bs
        
        ln = int.from_bytes(self.bs[4:5], byteorder='little')
        if ln != len(self.bs):
            print("Warning: ln != len(self.bs)")
            return self.bs

        fnc = self.bs[6]
        if fnc not in [0x31,0x32]:
            return self.bs

        img_format = self.bs[7:10].decode(encoding='utf8')
        if img_format.upper() not in ['JPG','BMP','PNG']:
            return self.bs

        img_size = int.from_bytes(self.bs[10:13], byteorder='little')
        if img_size <=0 :
            return self.bs
        
        img_path = self.bs[14:-1].decode(encoding='utf8')

        return fnc,img_format,img_size,img_path


class ClientProtocol(object):
    """
    规定：
        数据包头部占4字节
        报文总长度占2字节
        功能码占1字节
        数据域不定长
        结束符占1字节
        高字节在前，低字节在后
    """
    def __init__(self, bs=None):
        """
        如果bs为None则代表需要创建一个数据包
        否则代表需要解析一个数据包
        """
        if bs:
            self.bs = bytearray(bs)
        else:
            self.bs = bytearray(0)
    
    def encode(self,fnc,img_format,img_size,img_path):
        self.bs = bytearray(0)

        head = 'JGIR'
        self.bs += bytearray(head.encode(encoding='utf8'))

        # len: head(4) + ln(2) + fnc(1)  + format(3) + img_len(4) + img_path(n) + end(1)
        ln = (len(head.encode(encoding='utf8')) + 2 + 1 + 3 + 4 + len(img_path.encode(encoding='utf8')) + 1)

        if len(img_format.encode(encoding='utf8')) != 3:
            return self.bs

        self.bs += (bytearray(ln.to_bytes(2, byteorder='little')) +
                    bytearray(fnc.to_bytes(1, byteorder='little')))

        self.bs += (bytearray(img_format.encode(encoding='utf8')) +
                   bytearray(img_size.to_bytes(4, byteorder='little')) +
                   bytearray(img_path.encode(encoding='utf8')) +
                   bytearray((0x0a).to_bytes(1, byteorder='little')))

        return self.bs

    def decode(self,bs=None):
        if bs:
            ln = int.from_bytes(bs[4:5], byteorder='little')
            if ln != len(bs):
                print("Warning: ln != len(self.bs)")
                self.bs,_ = codecs.escape_decode(bs,'hex')
            else:
                self.bs = bs
        
        head = self.bs[:4].decode(encoding='utf8')
        if head != 'JGIR':
            return self.bs
        
        end = self.bs[-1]
        if end != 0x0a:
            return self.bs
        
        ln = int.from_bytes(self.bs[4:5], byteorder='little')
        if ln != len(self.bs):
            return self.bs

        fnc = self.bs[6]
        if fnc not in [0x31,0x32]:
            return self.bs

        results_num = self.bs[7]
        results = []
        for i in range(results_num):
            results.append(OrderedDict({
                'status':self.bs[8+i*34:10+i*34].decode(encoding='utf8'),
                'score':struct.unpack('f',self.bs[10+i*34:14+i*34]),
                'text':self.bs[14+i*34:25+i*34].decode(encoding='utf8'),
                'center_x':struct.unpack('f',self.bs[25+i*34:29+i*34]),
                'center_y':struct.unpack('f',self.bs[29+i*34:33+i*34]),
                'width':struct.unpack('f',self.bs[33+i*34:37+i*34]),
                'height':struct.unpack('f',self.bs[37+i*34:41+i*34])}))

        img_format = self.bs[42+i*34:45+i*34].decode(encoding='utf8')
        if img_format.upper() not in ['JPG','BMP','PNG']:
            return self.bs

        img_size = int.from_bytes(self.bs[45+i*34:49+i*34], byteorder='little')
        if img_size <=0 :
            return self.bs
        
        img_path = self.bs[49+i*34:-1].decode(encoding='utf8')

        return fnc,results,img_format,img_size,img_path
 
if __name__ == '__main__':
    sp = ServerProtocol()
    cp = ClientProtocol()
    fnc = 0x32
    results = [OrderedDict({'status':'OK',
                'score':0.9873022,
                'text':'0J049660904',
                'center_x':1607.5,
                'center_y':758.5,
                'width':1109.0,
                'height':141.0}),
                OrderedDict({'status':'NG',
                'score':0.9873022,
                'text':'0J049660904',
                'center_x':1607.5,
                'center_y':758.5,
                'width':1109.0,
                'height':141.0})]
    image_file = os.path.join('test.JPG')
    img = cv2.imread(image_file)    
    server_encode_msg = sp.encode(fnc,results,image_file[-3:],img.size,os.path.abspath(image_file))
    client_decode_msg = cp.decode(server_encode_msg)
    client_encode_msg = cp.encode(fnc,image_file[-3:],img.size,os.path.abspath(image_file))
    print(client_encode_msg)
    server_decode_msg = sp.decode(client_encode_msg)
    pass