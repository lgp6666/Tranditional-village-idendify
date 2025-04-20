import torch
from Model.Public_Modules.utils import evaluate
from Model.Public_Modules.my_dataset import MyDataSet
import Config.pytorch as pytorch_model
from Config.get_config import load_config
import warnings

warnings.filterwarnings('ignore')


def test(val_data_path, model_path, device, batch_size, nw, model_name):
    val_images_path, val_images_label = pytorch_model.pytorch_read(val_data_path)
    img_size = 224

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=pytorch_model.test_data(img_size))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = torch.load(model_path, map_location=torch.device(device)).to(device)

    # validate
    evaluate(model=model,
             data_loader=val_loader,
             device=device,
             epoch=0,
             model_name=model_name)


if __name__ == '__main__':
    # 填配置文件里的模型的键名:
    config_model_name = 'Model_RegNet'

    config = load_config()

    # 测试集所在的路径
    dir_path = config['project_path'] + '/' + config['test_data_path']

    # 训练的批次大小
    batch_size = config['batch_size']

    # 训练好的模型路径
    model_path = config['project_path'] + '/' + config[config_model_name]

    # number of workers 使用多少个线程加载数据
    nw = config['nw']

    # 使用什么机器训练模型
    device = config['device']

    test(dir_path, model_path, device, batch_size, nw, config_model_name)
