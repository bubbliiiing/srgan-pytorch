import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg19

from nets.srgan import Discriminator, Generator
from utils.dataloader import SRGAN_dataset_collate, SRGANDataset
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda            = True
    #-----------------------------------#
    #   代表进行四倍的上采样
    #-----------------------------------#
    scale_factor    = 4
    #-----------------------------------#
    #   获得输入与输出的图片的shape
    #-----------------------------------#
    lr_shape        = [96, 96]
    hr_shape        = [lr_shape[0] * scale_factor, lr_shape[1] * scale_factor]
    #--------------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    #--------------------------------------------------------------------------#
    G_model_path    = ""
    D_model_path    = ""

    #------------------------------#
    #   训练参数设置
    #------------------------------#
    Init_epoch      = 0
    Epoch           = 200
    batch_size      = 4
    lr              = 0.0002
    #------------------------------#
    #   每隔50个step保存一次图片
    #------------------------------#
    save_interval   = 50
    #------------------------------#
    #   获得图片路径
    #------------------------------#
    annotation_path = "train_lines.txt"

    #---------------------------#
    #   生成网络和评价网络
    #---------------------------#
    G_model = Generator(scale_factor)
    D_model = Discriminator()
    #-----------------------------------#
    #   创建VGG模型，该模型用于提取特征
    #-----------------------------------#
    VGG_model = vgg19(pretrained=True)
    VGG_feature_model = nn.Sequential(*list(VGG_model.features)[:-1]).eval()
    for param in VGG_feature_model.parameters():
        param.requires_grad = False

    #------------------------------------------#
    #   将训练好的模型重新载入
    #------------------------------------------#
    if G_model_path != '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = G_model.state_dict()
        pretrained_dict = torch.load(G_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        G_model.load_state_dict(model_dict)
    if D_model_path != '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = D_model.state_dict()
        pretrained_dict = torch.load(D_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        D_model.load_state_dict(model_dict)

    G_model_train = G_model.train()
    D_model_train = D_model.train()
    
    if Cuda:
        cudnn.benchmark = True
        G_model_train = torch.nn.DataParallel(G_model)
        G_model_train = G_model_train.cuda()

        D_model_train = torch.nn.DataParallel(D_model)
        D_model_train = D_model_train.cuda()

        VGG_feature_model = torch.nn.DataParallel(VGG_feature_model)
        VGG_feature_model = VGG_feature_model.cuda()

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()

    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        epoch_step      = min(num_train // batch_size, 2000)
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------#
        #   Adam optimizer
        #------------------------------#
        G_optimizer     = optim.Adam(G_model_train.parameters(), lr=lr, betas=(0.9, 0.999))
        D_optimizer     = optim.Adam(D_model_train.parameters(), lr=lr, betas=(0.9, 0.999))
        G_lr_scheduler  = optim.lr_scheduler.StepLR(G_optimizer,step_size=1,gamma=0.98)
        D_lr_scheduler  = optim.lr_scheduler.StepLR(D_optimizer,step_size=1,gamma=0.98)

        train_dataset   = SRGANDataset(lines, lr_shape, hr_shape)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=SRGAN_dataset_collate)

        for epoch in range(Init_epoch, Epoch):
            fit_one_epoch(G_model_train, D_model_train, G_model, D_model, VGG_feature_model, G_optimizer, D_optimizer, BCE_loss, MSE_loss, epoch, epoch_step, gen, Epoch, Cuda, batch_size, save_interval)
            G_lr_scheduler.step()
            D_lr_scheduler.step()
