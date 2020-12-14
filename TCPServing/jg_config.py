import os
cfg = {
    # Server config
    'host':'127.0.0.1',
    'port':1024,
    'buf_size':10*1024*1024*8,  #10 MB

    # Engine config
    'use_gpu':False,
    'gpu_mem':4000,
    'use_tensorrt':False,
    'enable_mkldnn':False,

    # Detection config
    'det_model_dir':os.path.join("..","PaddleOCR","inference","mobile_det_tsbd_slim",""),

    # Classification config
    'use_angle_cls':False,

    # Recognition config
    'rec_model_dir':os.path.join("..","PaddleOCR","inference","server_rec_tsbd_slim",""),
    'rec_char_dict_path':os.path.join("..","PaddleOCR","ppocr","utils","tsbd_dict.txt"),
    'rec_image_shape':"3, 38, 266",
    'drop_score':0.5
}