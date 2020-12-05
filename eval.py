from paddleocr import PaddleOCR # use tools.infer.predict_system instead
import edit_distance
import sys
import os
import time
import cv2
import numpy as np
import json
from utility import Logger,cal_time,cal_iou,get_size

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
        if not os.path.exists(os.path.join(output_dir,'visualization','incorrect')):
            os.makedirs(os.path.join(output_dir,'visualization','incorrect'))
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    sys.stdout = Logger(log_file_name, sys.stdout)
    sys.stderr = Logger(log_file_name, sys.stderr)
    wrong_rec_logger = []
    
    # Load model
    use_gpu = False 
    enable_mkldnn = True
    use_angle_cls = False   
    det_model_dir = os.path.join('PaddleOCR','inference','server_det_tsbd_slim')
    cls_model_dir = os.path.join('PaddleOCR','inference','ch_ppocr_mobile_v1.1_cls_infer')
    rec_model_dir = os.path.join('PaddleOCR','inference','server_rec_tsbd_slim')
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
                cv2.imwrite(output_dir+"visualization/incorrect/"+img_path,predImg)
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
                    cv2.imwrite(output_dir+"visualization/incorrect/"+img_path,predImg)
        AED = AED + NED
        
    AED = AED / (len(lines))
    P = P / (len(lines))
    IOU = IOU / (len(lines))
    
    # Calculate inference time
    total_predict_time,avg_time,avg_det_time,avg_rec_time = cal_time(log_file_name)

    # Print performance in log file
    print("\n\nDetection Model:{}({:.2f}MB)".format(det_model_dir,float(get_size(det_model_dir))/1024/1024))
    if(use_angle_cls):
        print("Classification Model:{}({:.2f}MB)".format(cls_model_dir,float(get_size(cls_model_dir))/1024/1024))
    print("Recognition Model:{}({:.2f}MB)".format(rec_model_dir,float(get_size(rec_model_dir))/1024/1024))
    if use_gpu:
        print("Use GPU("+os.environ["CUDA_VISIBLE_DEVICES"]+")") if "CUDA_VISIBLE_DEVICES" in os.environ else print("Use GPU(0)")
    else:
        print("Use CPU(enable mlkdnn)") if enable_mkldnn else print("Use CPU")            
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
        print("pred :",wrong_rec['pred'])
    print("Finished!")

