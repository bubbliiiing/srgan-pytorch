#--------------------------------------------------------------#
#   对单张图片进行预测，运行结果保存在根目录
#   默认保存文件为results/predict_out/predict_srgan.png
#--------------------------------------------------------------#
from PIL import Image

from srgan import SRGAN

if __name__ == "__main__":
    srgan = SRGAN()
    #----------------------------#
    #   单张图片的保存路径
    #----------------------------#
    save_path_1x1 = "results/predict_out/predict_srgan.png"

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = srgan.generate_1x1_image(image)
            r_image.save(save_path_1x1)
            r_image.show()
