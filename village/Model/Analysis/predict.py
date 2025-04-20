import os
import torch
from PIL import Image
import Config.pytorch as pytorch_model
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
        conf = format(conf, ".4f")
    return predict_cla, conf


if __name__ == '__main__':
    # 填配置文件里的模型的键名:
    config_model_name = 'Improve_ConvNeXt'

    config = load_config()

    # 识别的类别数量
    num_classes = config['num_classes']

    # 训练好的模型路径
    model_path = config['project_path'] + '/' + config[config_model_name]

    # 使用什么机器训练模型
    device = config['device']

    # 待推理的图片路径
    img_path = config['project_path'] + '/' + config['img_path']

    model = torch.load(model_path, map_location=torch.device(device)).to(device)
    model.eval()

    # 预测
    result, confidence = predict_img(img_path, model, device)
    print('检测的图片路径：' + img_path)
    print('所属类别序号:' + str(result))
    print('置信度:' + str(confidence))
