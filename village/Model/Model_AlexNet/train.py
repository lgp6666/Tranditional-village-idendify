import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import csv
from Model.Public_Modules.my_dataset import MyDataSet
from Model.Public_Modules.utils import train_one_epoch, evaluate
from Model.Model_AlexNet.model import create_AlexNet as create_model
import Config.pytorch as pytorch_model
from Config.get_config import load_config
from torchvision import models
import math
import warnings
warnings.filterwarnings('ignore')


def train(device, num_classes, train_data_path, val_data_path,
          epochs, batch_size, weights_path, opt_name, nw, model_name, fine_tuning):
    # time_dir是日志文件文件保存时候的前缀数字，每次训练的时候都不同，这样不会覆盖之前的
    name_num = int(time.time())
    time_dir = name_num * 25

    # 检查CSV文件夹是否存在
    if not os.path.exists('CSV'):
        # 如果文件夹不存在，则创建它
        os.makedirs('CSV')
    # 检查logs文件夹是否存在
    if not os.path.exists('logs'):
        # 如果文件夹不存在，则创建它
        os.makedirs('logs')
    # 检查weights文件夹是否存在
    if not os.path.exists('weights'):
        # 如果文件夹不存在，则创建它
        os.makedirs('weights')
    # 检查result文件夹是否存在
    if not os.path.exists('result'):
        # 如果文件夹不存在，则创建它
        os.makedirs('result')

    with open('CSV\\' + model_name + '_' + str(time_dir) + '_metrics.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1',
                             'Valid Loss', 'Valid Accuracy', 'Valid Precision', 'Valid Recall', 'Valid F1'])

        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

        tb_writer = SummaryWriter(log_dir="logs/" + model_name + '_' + str(time_dir))

        train_images_path, train_images_label = pytorch_model.pytorch_read(train_data_path)
        val_images_path, val_images_label = pytorch_model.pytorch_read(val_data_path)

        img_size = 224

        # 实例化训练数据集
        train_dataset = MyDataSet(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=pytorch_model.train_data(img_size))

        # 实例化验证数据集
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=pytorch_model.val_data(img_size))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)

        if fine_tuning:
            model = models.alexnet(weights=None)
            # 训练集所在的路径（可以用绝对路径也可以相对路径）
            pre_model_path = config['project_path'] + '/Config/pretrained/alexnet_pre.pth'
            model.load_state_dict(torch.load(pre_model_path))
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

            # 解冻所有BN层
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = True
            # # 解冻部分特征层
            params = list(model.parameters())
            num_layers_to_unfreeze = math.ceil(len(params) / 20)
            for param in params[-num_layers_to_unfreeze:]:
                param.requires_grad = True
        else:
            model = create_model(model_name=model_name, num_classes=num_classes)
            if weights_path != "":
                assert os.path.exists(weights_path), "weights file: '{}' not exist.".format(weights_path)
                model = torch.load(weights_path)
            else:
                model.apply(pytorch_model.weights_init)

        # inplace设置为False
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        print(model)
        # 打印所有层的冻结或解冻状态
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Frozen: {not param.requires_grad}")
        model = model.to(device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        print(f'Trainable parameters: {trainable_params}')
        print(f'Frozen parameters: {frozen_params}')
        print(f'ALL parameters: {frozen_params + trainable_params}')

        if opt_name == 'SGD':
            lr = 0.001
            wd = 0
            pg = pytorch_model.pytorch_params(model, weight_decay=wd)
            optimizer = optim.SGD(
                pg,
                lr=lr,  # 学习率
                momentum=0,  # 动量因子
                dampening=0,  # 动量抑制因子
                weight_decay=0,  # 权重衰减（L2正则化）
                nesterov=False  # 是否使用Nesterov动量
            )
        elif opt_name == 'Adam':
            lr = 0.001
            wd = 0
            pg = pytorch_model.pytorch_params(model, weight_decay=wd)
            optimizer = optim.Adam(
                pg,
                lr=lr,  # 学习率
                betas=(0.9, 0.999),  # 用于计算梯度和梯度平方的运行平均值的系数
                eps=1e-08,  # 为了数值稳定性而加到分母里的项
                weight_decay=wd,  # 权重衰减（L2正则化）
                amsgrad=False  # 是否使用AMSGrad变种
            )
        elif opt_name == 'RMSprop':
            lr = 0.01
            wd = 0
            pg = pytorch_model.pytorch_params(model, weight_decay=wd)
            optimizer = optim.RMSprop(
                pg,
                lr=lr,  # 学习率
                alpha=0.99,  # 梯度平方的移动平均系数
                eps=1e-08,  # 为了数值稳定性而加到分母里的项
                weight_decay=wd,  # 权重衰减（L2正则化）
                momentum=0,  # 动量因子
                centered=False  # 是否计算中心化的二阶矩
            )
        elif opt_name == 'Adagrad':
            lr = 0.01
            wd = 0
            pg = pytorch_model.pytorch_params(model, weight_decay=wd)
            optimizer = optim.Adagrad(
                pg,
                lr=lr,  # 学习率
                lr_decay=0,  # 学习率衰减
                weight_decay=wd,  # 权重衰减（L2正则化）
                initial_accumulator_value=0,  # 初始累加值
                eps=1e-10  # 为了数值稳定性而加到分母里的项
            )
        elif opt_name == 'AdamW':
            lr = 0.001
            wd = 0.01
            pg = pytorch_model.pytorch_params(model, weight_decay=wd)
            optimizer = optim.AdamW(
                pg,
                lr=lr,  # 学习率
                betas=(0.9, 0.999),  # 用于计算梯度和梯度平方的运行平均值的系数
                eps=1e-08,  # 为了数值稳定性而加到分母里的项
                weight_decay=wd,  # 权重衰减（L2正则化）
                amsgrad=False  # 是否使用AMSGrad变种
            )
        else:
            return None

        lr_scheduler = pytorch_model.pytorch_scheduler(optimizer, len(train_loader), epochs,
                                           warmup=True, warmup_epochs=1)

        best_acc = 0.

        for epoch in range(epochs):
            # train
            train_loss, train_acc, train_P, train_R, train_F1 = train_one_epoch(model=model,
                                                                                optimizer=optimizer,
                                                                                data_loader=train_loader,
                                                                                device=device,
                                                                                epoch=epoch,
                                                                                lr_scheduler=lr_scheduler,
                                                                                epochs=epochs)

            # validate
            val_loss, val_acc, val_P, val_R, val_F1 = evaluate(model=model,
                                                               data_loader=val_loader,
                                                               device=device,
                                                               epoch=epoch)

            tags = ["train_loss", "train_acc",
                    "train_P", "train_R", "train_F1",
                    "val_loss", "val_acc",
                    "val_P", "val_R", "val_F1",
                    "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], train_P, epoch)
            tb_writer.add_scalar(tags[3], train_R, epoch)
            tb_writer.add_scalar(tags[4], train_F1, epoch)
            tb_writer.add_scalar(tags[5], val_loss, epoch)
            tb_writer.add_scalar(tags[6], val_acc, epoch)
            tb_writer.add_scalar(tags[7], val_P, epoch)
            tb_writer.add_scalar(tags[8], val_R, epoch)
            tb_writer.add_scalar(tags[9], val_F1, epoch)
            # tb_writer.add_scalar(tags[10], optimizer.param_groups[0]["lr"], epoch)

            # 写入CSV文件
            csv_writer.writerow(
                [epoch, train_loss, train_acc, train_P, train_R, train_F1, val_loss, val_acc,
                 val_P, val_R, val_F1])

            if best_acc < val_acc:
                torch.save(model, "./weights/" + model_name + '_' + str(time_dir) + "_best_model.pt")
                best_acc = val_acc


if __name__ == '__main__':
    model_name = 'AlexNet'
    config = load_config()

    # 分类的类别数量
    num_classes = config['num_classes']

    # 训练集所在的路径（可以用绝对路径也可以相对路径）
    train_data_path = config['project_path'] + '/' + config['train_data_path']

    # 验证集所在的路径（可以用绝对路径也可以相对路径）
    val_data_path = config['project_path'] + '/' + config['val_data_path']

    # 训练的轮次
    epochs = config['epochs']

    # 训练的批次大小
    batch_size = config['batch_size']

    # 模型路径
    weights_path = config['weights_path']
    if weights_path != '':
        weights_path = config['project_path'] + '/' + config['weights_path']

    # 优化器
    opt_name = config['opt_name']

    # 使用什么机器训练模型
    device = config['device']

    # number of workers 使用多少个线程加载数据
    nw = config['nw']

    fine_tuning = config['fine_tuning']

    train(device, num_classes, train_data_path, val_data_path, epochs,
          batch_size, weights_path, opt_name, nw, model_name,fine_tuning)

