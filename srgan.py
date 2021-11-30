import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.srgan import Generator
from utils.utils import cvtColor, preprocess_input


class SRGAN(object):
    #-----------------------------------------#
    #   注意修改model_path
    #-----------------------------------------#
    _defaults = {
        #-----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------#
        "model_path"        : 'model_data/Generator_SRGAN.pth',
        #-----------------------------------------------#
        #   上采样的倍数，和训练时一样
        #-----------------------------------------------#
        "scale_factor"      : 4, 
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   初始化SRGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
        self.generate()

    def generate(self):
        self.net    = Generator(self.scale_factor)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
        
    def generate_1x1_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，并进行归一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image, dtype='float32'), [0.5,0.5,0.5], [0.5,0.5,0.5]), [2,0,1]), 0)
        
        with torch.no_grad():
            image_data = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                image_data = image_data.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            hr_image = self.net(image_data)[0]
            #---------------------------------------------------------#
            #   将归一化的结果再转成rgb格式
            #---------------------------------------------------------#
            hr_image = (hr_image.cpu().data.numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
            hr_image = Image.fromarray(np.uint8(hr_image))

            return hr_image
