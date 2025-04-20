import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from Model.Model_Swin_Transformer.GradCAM_utils import GradCAM, show_cam_on_image, center_crop_img
import Model.Analysis.predict as predict
import warnings
warnings.filterwarnings('ignore')
import time
from Config.get_config import load_config
import math


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)
        self.channels = 768  # Swin-Transformer 的输出通道数

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        # print(f"Original shape: {x.shape}")
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           self.channels)
        # print(f"Reshaped to: {result.shape}")

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        # print(f"Permuted to: {result.shape}")

        return result


def visualize_image(model_weight_path, device, img_path, model_name, show=True):
    model = torch.load(model_weight_path, map_location=torch.device(device)).to(device)
    model.eval()
    target_category, conf = predict.predict_img(img_path, model, device)
    target_category = int(target_category)
    target_layers = [model.norm]

    # 注意输入的图片必须是32的整数倍
    # 否则由于padding的原因会出现注意力飘逸的问题
    img_size = 224
    assert img_size % 32 == 0

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, img_size)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)

    now = int(time.time()) * 25
    save_dir = './result/Grad_CAM_' + model_name + '_' + str(now) + '.png'
    plt.savefig(save_dir, dpi=500, bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':
    config_model_name = 'Model_Swin_Transformer'
    config = load_config()

    # 训练好的权重路径
    model_weight_path = config['project_path'] + '/' + config[config_model_name]

    # 待可视化的图片
    img_path = config['project_path'] + '/' + config['img_path']

    # 使用什么机器训练模型
    device = config['device']

    visualize_image(model_weight_path, device, img_path, config_model_name)


