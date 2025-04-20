import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from Model.Model_MobileNet.GradCAM_utils import GradCAM, show_cam_on_image
import time
import Model.Analysis.predict as predict
import Config.pytorch as pytorch_model
from Config.get_config import load_config
import warnings
warnings.filterwarnings('ignore')


def visualize_image(model_weight_path, device, img_path, model_name, show=True):
    model = torch.load(model_weight_path, map_location=torch.device(device)).to(device)
    model.eval()
    target_category, conf = predict.predict_img(img_path, model, device)
    target_category = int(target_category)
    # target_layers = [model.features[-1]]
    target_layers = [model.features[-1]]
    for param in model.parameters():
        param.requires_grad = True

    data_transform = pytorch_model.norm_data()
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # 确保图像尺寸为 224x224
    img = np.array(img, dtype=np.uint8)

    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    now = int(time.time()) * 520
    save_dir = './result/Grad_CAM_' + model_name + '_' + str(now) + '.png'
    plt.savefig(save_dir, dpi=500, bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':
    config_model_name = 'Model_MobileNet'
    config = load_config()

    # 识别的类别数量
    num_classes = config['num_classes']

    # 训练好的模型路径
    model_path = config['project_path'] + '/' + config[config_model_name]

    # 待可视化的图片
    img_path = config['project_path'] + '/' + config['img_path']

    # 使用什么机器训练模型
    device = config['device']

    visualize_image(model_path, device, img_path, config_model_name)

