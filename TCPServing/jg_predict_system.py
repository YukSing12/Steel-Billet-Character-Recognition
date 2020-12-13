# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..','PaddleOCR',)))


import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger

logger = initial_logger()
import cv2
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
import copy
import numpy as np
import math
import time
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from PIL import Image
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt
from jg_protocol import ServerProtocol
import socket

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        # print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            # print("cls num  : {}, elapse : {}".format(
                # len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        # print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(**kwargs):
    args = utility.parse_args()
    args.__dict__.update(**kwargs)
    text_sys = TextSystem(args)
    img = cv_imread("test.JPG")
    if img is None:
        logger.info("error in loading image:{}".format(image_file))
    dt_boxes, rec_res = text_sys(img)

    logger.info("Finished initlizing recognition system.")
    is_visualize = False
    font_path = args.vis_font_path

    # Start server system
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # host = '127.0.0.1'
    host = '127.0.0.1'
    port = 1024
    buf_size = 10*1024*1024*8

    server_socket.bind((host, port))
    server_socket.listen(5)
    sp = ServerProtocol()
    logger.info("Started server system.")
    logger.info("Host:{}, port:{}, recieve buf size:{}".format(host,port,buf_size))
    while True:
        logger.info("Wait for connection.")
        clientsocket,addr = server_socket.accept()

        logger.info("Connected IP Address: %s" % str(addr))
        clientsocket.send('Server of steel billet recognition system\r\n'.encode('utf-8'))
        
        while True: # 与客户端阻塞式通信
            try:
                msg = clientsocket.recv(buf_size)
            except ConnectionResetError as err:
                logger.error("Connection Reset Error:{}".format(err))
                break
            except:
                logger.error("Unexpected Error")
                break
            
            if not len(msg):
                break
            fnc,img_format,img_size,image_file = sp.decode(msg)
            logger.info('fnc:0x{:x}, img_format:{}, img_size:{}, img_path:{}'.format(fnc,img_format,img_size,image_file))

            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv_imread(image_file)
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            starttime = time.time()
            dt_boxes, rec_res = text_sys(img)
            elapse = time.time() - starttime
            logger.info("Predict time of %s: %.3fs" % (os.path.abspath(image_file), elapse))

            drop_score = 0.5
            dt_num = len(dt_boxes)
            results = []
            for dno in range(dt_num):
                text, score = rec_res[dno]
                center_x = float(np.mean(dt_boxes[dno][:,0]))
                center_y = float(np.mean(dt_boxes[dno][:,1]))
                width = np.max(dt_boxes[dno][:,0])-np.min(dt_boxes[dno][:,0])
                height = np.max(dt_boxes[dno][:,1])-np.min(dt_boxes[dno][:,1])
                result_dict = {'status':'OK' if score >= drop_score else 'NG',
                                'score':score,
                                'text':text,
                                'center_x':center_x,
                                'center_y':center_y,
                                'width':width,
                                'height':height,
                                }
                results.append(result_dict)
            print("Encode:")
            print(sp.encode(0x32,results,image_file[-3:],img.size,os.path.abspath(image_file)))
            clientsocket.send(sp.bs)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path)
                draw_img_save = "./inference_results/"
                if not os.path.exists(draw_img_save):
                    os.makedirs(draw_img_save)
                cv2.imwrite(
                    os.path.join(draw_img_save, os.path.basename(image_file)),
                    draw_img[:, :, ::-1])
                print("The visualized image saved in {}".format(
                    os.path.join(draw_img_save, os.path.basename(image_file))))

        clientsocket.close()
        logger.info("Connection %s closed" % str(addr))


if __name__ == "__main__":
    main(use_gpu = False,
        enable_mkldnn = False,
        use_angle_cls = False,
        det_model_dir = os.path.join("..","tmp","Steel-Billet-Character-Recognition","PaddleOCR","inference","mobile_det_tsbd_slim",""),
        rec_model_dir = os.path.join("..","tmp","Steel-Billet-Character-Recognition","PaddleOCR","inference","server_rec_tsbd_slim",""),
        rec_char_dict_path = os.path.join("..","tmp","Steel-Billet-Character-Recognition","PaddleOCR","ppocr","utils","tsbd_dict.txt"),
        rec_image_shape = "3, 38, 266")
