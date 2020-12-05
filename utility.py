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
    """ Calculates iou(intersection-over unit) between box1 and box2.
    
    Args:
        box1(np.array([[ptx1,pty1],[ptx2,pty2],[ptx3,pty3],[ptx4,pty4]]))
        box2(np.array([[ptx5,pty5],[ptx6,pty6],[ptx7,pty7],[ptx8,pty8]]))

    Returns: 
        iou: iou(intersection-over unit) between box1 and box2.
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
    """ Calculates inference time of difference stages in the model through a log file.

    Args:
        log_file(str): Name of log file.

    Returns: 
        total_predict_time: Total inference time(s) of entire model.
        avg_time: Average inference time(ms) of entire model.
        avg_det_time: Average inference time(ms) of detection model.
        avg_rec_time: Average inference time(ms) of recognitin model.
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
                time = time.replace('\n','')
                time = time.replace('s','')
                total_predict_time = (total_predict_time + float(time))
                count = count + 1
    #print("count:",count)
    avg_det_time = avg_det_time / count    
    avg_rec_time = avg_rec_time / count
    avg_time = total_predict_time / count
    return total_predict_time,avg_time,avg_det_time,avg_rec_time

def get_size(file_path):
    """ Get size of file or directory.

    Args:
        file_path(str): Path of file or directory.

    Returns: 
        size(int): Size of file or directory in bits.
    """
    size = 0
    if os.path.isdir(file_path):
        for root, dirs, files in os.walk(file_path):
            for f in files:
                size += os.path.getsize(os.path.join(root, f))
    elif os.path.isfile(file_path):
        size = (os.path.getsize(file_path))
    return size