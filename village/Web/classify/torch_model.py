import Config.pytorch as pytorch_model
import torch
from PIL import Image
import os
from Config.get_config import load_config
import warnings

warnings.filterwarnings('ignore')


def predict_img(img_path, model, device):
    img_size = 224
    data_transform = pytorch_model.val_data(img_size)
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0).to(device)

    with torch.no_grad():
        output = torch.squeeze(model(img)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        conf = predict[predict_cla].item()
    return predict_cla, conf


class GlobalObject:
    _instance = None

    def __init__(self):
        # print("预加载模型到GPU")
        config = load_config()
        # 识别的类别数量
        num_classes = config['num_classes']

        # 训练好的权重路径
        model_weight_path = config['project_path'] + '/' + config['Web_Model']

        device = config['device']

        model = torch.load(model_weight_path, map_location=torch.device(device)).to(device)
        model.eval()
        predict_img('./static/img/bk1.jpg', model, device)

        self.some_property = model
