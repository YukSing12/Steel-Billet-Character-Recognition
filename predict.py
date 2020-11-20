from paddleocr import PaddleOCR
import edit_distance
import sys
import os
import time
import cv2
import numpy as np
import json

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


def cal_iou(box1, box2):
    """
    :param box1: = np.array([[ptx1,pty1],[ptx2,pty2],[ptx3,pty3],[ptx4,pty4]])
    :param box2: = np.array([[ptx5,pty5],[ptx6,pty6],[ptx7,pty7],[ptx8,pty8]])
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = min(box1[:,0]),min(box1[:,1]),max(box1[:,0]),max(box1[:,1])
    xmin2, ymin2, xmax2, ymax2 = min(box2[:,0]),min(box2[:,1]),max(box2[:,0]),max(box2[:,1])

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  
 
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h 
    iou = area / (s1 + s2 - area)
    return iou


if __name__=='__main__':

    # Preference
    output_dir = "./output/"
    visualization = True
    img_set_dir = "Tangshan-Steel-Billet-Dataset/det_image/test/"
    #img_set_dir = "Steel-Billet-Dataset/test_image/"

    label_file = "Tangshan-Steel-Billet-Dataset/text_localization_test_label.txt"
    #label_file = "Steel-Billet-Dataset/text_localization_test_label.txt"
    
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Init Logger
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    t = 'img2'
    output_dir = output_dir + str(t) + "/"
    if visualization:
        if not os.path.exists(output_dir+"visualization/"):
            os.makedirs(output_dir+"visualization/")
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    sys.stdout = Logger(output_dir+str(t)+".log", sys.stdout)
    sys.stderr = Logger(output_dir+str(t)+"_error.log", sys.stderr)
    wrong_rec_logger = []

    # Load model
    use_gpu = True
    enable_mkldnn = False
    use_angle_cls = False   
    det_model_dir = "PaddleOCR/inference/mobile_det_tsbd2"
    cls_model_dir = "PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer"
    rec_model_dir = "PaddleOCR/inference/server_rec_tsbd2"
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang="ch",use_gpu=use_gpu,use_space_char=False,gpu_mem=8000,
                    enable_mkldnn = enable_mkldnn,
                    rec_char_dict_path = "/home/aistudio/PaddleOCR/ppocr/utils/tsbd_dict.txt",         
                    rec_image_shape = "3, 38, 266",
                    det_model_dir=det_model_dir,cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir
                    )

    # Average Edit-Distance
    AED = 0
    # Word-level precision
    P = 0
    IOU = 0
    count = 0

    # Visualization of model
    
    print("Visualization:",visualization)        

    # Load data in a folder
    with open(label_file) as test_data:
        lines = test_data.readlines()

    #Inference in a folder
    sys.stdout = open(os.devnull, 'w')
    for line in lines:
        # Decode
        line = line.replace("\n","")
        [img_path,img_label] = line.split("\t")
        img_label = json.loads(img_label)
        img_box = img_label[0]['points']
        img_box = np.array(img_box,np.int32)
        img_label = img_label[0]['transcription']
        img_label = img_label.replace(" ","")
        
        result = ocr.ocr(img_set_dir+img_path, cls=use_angle_cls)
        # Calculate word-level precision, NED
        NED = 1
        if visualization:
            predImg = cv2.imread(img_set_dir+img_path)
        if len(result)==0:
            count+=1
            wrong_rec_logger.append({'path':img_path,'label':img_label,'pred':None})
            if visualization:           
                cv2.imwrite(output_dir+"visualization/wrong/"+img_path,predImg)
        for info in result:
            pred_box = info[0]
            pred_box = np.array(pred_box,np.int32)
            pred_label = info[1][0]
            pred_score = info[1][1]
            NED = min(NED,edit_distance.SequenceMatcher(img_label,pred_label).distance()/max(len(img_label),len(pred_label)))
            IOU = IOU + cal_iou(img_box,pred_box)
            if visualization:
                cv2.polylines(predImg, [pred_box], 1, (0,255,0))
                cv2.putText(predImg, 'confidence:'+str(pred_score), (pred_box[0][0],pred_box[0][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                cv2.putText(predImg, 'text:'+str(pred_label), (pred_box[0][0],pred_box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
            if pred_label==img_label:
                P = P + 1
                if visualization:           
                    cv2.imwrite(output_dir+"visualization/correct/"+img_path,predImg)
            else:
                wrong_rec_logger.append({'path':img_path,'label':img_label,'pred':pred_label})
                if visualization:           
                    cv2.imwrite(output_dir+"visualization/wrong/"+img_path,predImg)
        AED = AED + NED
        
    sys.stdout = Logger(output_dir+str(t)+".log", sys.stdout)
    AED = AED / (len(lines))
    P = P / (len(lines))
    IOU = IOU / (len(lines))

    # Calculate inference time
    print("Calculating inference time")
    duration = 0

    sys.stdout = open(os.devnull, 'w')
    start = time.process_time()
    for line in lines:
        [img_path,img_label] = line.split("\t")
        result = ocr.ocr(img_set_dir+img_path, cls=use_angle_cls)
    end = time.process_time()
    sys.stdout = Logger(output_dir+str(t)+".log", sys.stdout)

    # Print performance in log file
    print("Detection Model:"+det_model_dir)
    if(use_angle_cls):
        print("Classification Model:"+cls_model_dir)
    print("Recognition Model:"+rec_model_dir)
    if use_gpu:
        print("Use GPU")
    else:
        print("Use CPU(mlkdnn)")
    print("The number of all test images:"+str(len(lines)))
    print("The number of images failed the test:"+str(count))
    print("Detection intersection-over unit:"+str(IOU))
    print("Precision:"+str(P))
    print("Accuracy:"+str(1-AED))
    print("Average normalized edit-distance:"+str(AED))
    print("Inference time(s):"+str(end-start))
    print("Mean inference time(ms):"+str((end-start)/len(lines)*1000))
    print("\n")
    print("Wrong recognition details:")
    for wrong_rec in wrong_rec_logger:
        print(wrong_rec['path'])
        print("label:",wrong_rec['label'])
        print("pred:",wrong_rec['pred'])

    sys.stdout = sys.__stdout__
    with open(output_dir+str(t)+".log",'r') as log:
        lines = log.readlines()
        for line in lines:
            print(line.replace("\n",""))
