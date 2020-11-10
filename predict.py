from paddleocr import PaddleOCR
import edit_distance
import sys
import os
import time
import cv2
import numpy as np

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

if __name__=='__main__':

    # Preference
    output_dir = "./output/"
    visualization = False
    img_set_dir = "Tangshan-Steel-Billet-Dataset/det_image/eval/"
    label_file = "Tangshan-Steel-Billet-Dataset/text_recognition_eval_label.txt"
    use_gpu = True
    use_angle_cls = True   
    # det_model_dir = "PaddleOCR/inference/mobile_det_tsbd"
    # cls_model_dir = "PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer"
    # rec_model_dir = "PaddleOCR/inference/server_rec_tsbd"
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang="ch",use_gpu=use_gpu,use_space_char=False,gpu_mem=8000,
                    #enable_mkldnn = True,
                    #rec_char_dict_path = "/home/aistudio/PaddleOCR/ppocr/utils/tsbd_dict.txt",         
                    #rec_image_shape = "3, 38, 266",
                    #det_model_dir=det_model_dir,cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir
                    )

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Init Logger
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
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

    # Average Edit-Distance
    AED = 0
    # Word-level precision
    P = 0
    count = 0

    # Visualization of model
    
    print("Visualization:",visualization)        

    # Load data in a folder
    with open(label_file) as test_data:
        lines = test_data.readlines()

    #Inference in a folder
    sys.stdout = open(os.devnull, 'w')
    for line in lines:
        line = line.replace("\n","")
        [img_path,img_label] = line.split("\t")
        img_label = img_label.replace(" ","")

        result = ocr.ocr(img_set_dir+img_path, cls=use_angle_cls)
        # Calculate word-level precision, AED
        NED = 1
        if visualization:
            predImg = cv2.imread(img_set_dir+img_path)
        if len(result)==0:
            count+=1
            wrong_rec_logger.append({'path':img_path,'label':img_label,'pred':None})
        for info in result:
            pred_box = info[0]
            pred_label = info[1][0]
            pred_score = info[1][1]
            NED = min(NED,edit_distance.SequenceMatcher(img_label,pred_label).distance()/max(len(img_label),len(pred_label)))
            if pred_label==img_label:
                P = P + 1
            else:
                wrong_rec_logger.append({'path':img_path,'label':img_label,'pred':pred_label})
            if visualization:
                pred_box = np.array(pred_box,np.int32)
                cv2.polylines(predImg, [pred_box], 1, (0,255,0))
                cv2.putText(predImg, 'confidence:'+str(pred_score), (pred_box[0][0],pred_box[0][1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                cv2.putText(predImg, 'text:'+str(pred_label), (pred_box[0][0],pred_box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

        if visualization:           
            cv2.imwrite(output_dir+"visualization/"+img_path,predImg)
        AED = AED + NED
        
    sys.stdout = Logger(output_dir+str(t)+".log", sys.stdout)
    AED = AED / (len(lines))
    P = P / (len(lines))

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
    print("Use GPU:"+str(use_gpu))
    print("The number of all test images:"+str(len(lines)))
    print("The number of images failed the test:"+str(count))
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
            print(line)
