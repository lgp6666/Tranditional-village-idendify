from .apps import global_object_instance  # 引入已经初始化的全局对象
import Config.pytorch as pytorch_model
import torch
from PIL import Image
import os
from Config.get_config import load_config


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


def MyPrediction(picname):
    config = load_config()
    model = global_object_instance.some_property
    img_path = './media/pic/' + picname
    device = config['device']
    predict_cla, conf = predict_img(img_path, model, device)
    return predict_cla, conf

