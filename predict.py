from paddleocr import PaddleOCR # use tools.infer.predict_system instead
import paddle.fluid.profiler as profiler
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

def cal_time(log_file):
    """
    :param log_file: Name of log file.
    :return: 
        :total_predict_time: Total inference time(s) of entire model.
        :avg_time: Average inference time(ms) of entire model.
        :avg_det_time: Average inference time(ms) of detection model.
        :avg_rec_time: Average inference time(ms) of recognitin model.
    """
    print("Calculating inference time")
    log_file = os.path.join(log_file)
    count = 0
    total_predict_time = 0
    avg_det_time = 0
    avg_rec_time = 0
    with open(log_file,'r') as fid:
        lines = fid.readlines()
        for line in lines[5:]:
            if 'dt_boxes' in line:
                _,time = line.split('elapse :')
                time = time.replace('\n','')
                avg_det_time = avg_det_time + float(time)
            elif 'rec_res' in line:
                _,time = line.split('elapse :')
                time = time.replace('\n','')
                avg_rec_time = avg_rec_time + float(time)
            elif 'Predict time of' in line:
                _,time = line.split(': ')
                time = time.replace('s\n','')
                total_predict_time = (total_predict_time + float(time))
                count = count + 1
    avg_det_time = avg_det_time / count    
    avg_rec_time = avg_rec_time / count
    avg_time = total_predict_time / count
    return total_predict_time,avg_time,avg_det_time,avg_rec_time

if __name__=='__main__':

    # Preference
    output_dir = 'output'
    visualization = False
    img_set_dir = os.path.join('Tangshan-Steel-Billet-Dataset','det_image','eval','')
#    img_set_dir = os.path.join('Xiang-Steel-Billet-Dataset','test_image','')


    label_file = os.path.join('Tangshan-Steel-Billet-Dataset','text_localization_eval_label.txt')
    #label_file = os.path.join('Xiang-Steel-Billet-Dataset','text_localization_eval_label.txt')

    # Init Logger
    log_file_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#    log_file_name = 'mobile_det+server_rec_sbd'
    output_dir = os.path.join(output_dir,str(log_file_name),'')
    log_file_name = os.path.join(output_dir,str(log_file_name)+".log")
    if visualization:
        if not os.path.exists(os.path.join(output_dir,'visualization','correct')):
            os.makedirs(os.path.join(output_dir,'visualization','correct'))
        if not os.path.exists(os.path.join(output_dir,'visualization','wrong')):
            os.makedirs(os.path.join(output_dir,'visualization','wrong'))
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    sys.stdout = Logger(log_file_name, sys.stdout)
    sys.stderr = Logger(log_file_name, sys.stderr)
    wrong_rec_logger = []
    
    # Load model
    use_gpu = True 
    enable_mkldnn = False
    use_angle_cls = False   
    det_model_dir = os.path.join('PaddleOCR','inference','server_det_tsbd_slim')
    cls_model_dir = os.path.join('PaddleOCR','inference','ch_ppocr_mobile_v1.1_cls_infer')
    rec_model_dir = os.path.join('PaddleOCR','inference','server_rec_tsbd')
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang="ch",use_gpu=use_gpu,use_space_char=False,gpu_mem=4000,
                    enable_mkldnn = enable_mkldnn,
                    rec_char_dict_path = os.path.join('ppocr','utils','tsbd_dict.txt'), #Use specific dictionary         
                    rec_image_shape = "3, 38, 266",                                            #Use specific shape of recognition image
                    det_model_dir=det_model_dir,cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir
                    )

    # Average Edit-Distance
    AED = 0
    # Word-level precision
    P = 0
    # Intersection-over unit
    IOU = 0
    count = 0

    # Visualization of model
    
    print("Visualization:",visualization)        

    # Load data in a folder
    with open(label_file) as test_data:
        lines = test_data.readlines()

    #Inference in a folder
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
        # Calculate normalized-edit distance, NED
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
        
    AED = AED / (len(lines))
    P = P / (len(lines))
    IOU = IOU / (len(lines))

    # Calculate inference time
    total_predict_time,avg_time,avg_det_time,avg_rec_time = cal_time(log_file_name)

    # Print performance in log file
    print("\n\nDetection Model:"+det_model_dir)
    if(use_angle_cls):
        print("Classification Model:"+cls_model_dir)
    print("Recognition Model:"+rec_model_dir)
    if use_gpu:
        print("Use GPU")
    else:
        if enable_mkldnn:
            print("Use CPU(enable mlkdnn)")
        else:
            print("Use CPU")
    print("The number of all test images:"+str(len(lines)))
    print("The number of images failed the test:"+str(count))
    print("Detection intersection-over unit:"+str(IOU))
    print("Precision:"+str(P))
    print("Accuracy:"+str(1-AED))
    print("Average normalized edit-distance:"+str(AED))
    print("Total inference time(s) of entire model:"+str(total_predict_time))
    print("Average inference time(ms) of entire model:"+str(avg_time*1000))
    print("Average inference time(ms) of detection model:"+str(avg_det_time*1000))
    print("Average inference time(ms) of recognition model:"+str(avg_rec_time*1000))
    print("\n")
    print("Wrong recognition details:")
    for wrong_rec in wrong_rec_logger:
        print(wrong_rec['path'])
        print("label:",wrong_rec['label'])
        print("pred:",wrong_rec['pred'])
    print("Finished!")

