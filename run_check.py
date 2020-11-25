import os
import paddle.fluid as fluid 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#gpu_place = fluid.CUDAPlace(4) 
fluid.install_check.run_check()
