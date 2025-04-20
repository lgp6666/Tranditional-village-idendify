# -*- coding: UTF-8 -*-
import Config.pytorch as pytorch_model
import torch
import os
from Config.get_config import load_config
import warnings
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil
import threading
import openpyxl
from images.resizeAndPadding import pic_function
warnings.filterwarnings('ignore')


class MainWindow(QTabWidget):
    # 初始化
    def __init__(self):

        super().__init__()
        self.webcam = True
        self.vid_source = 0
        self.stopEvent = threading.Event()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('基于深度学习的花朵图像分类算法研究')
        # 模型初始化
        config = load_config()
        # 识别的类别数量
        num_classes = config['num_classes']

        # 训练好的权重路径
        model_weight_path = config['project_path'] + '/' + config['Web_Model']

        self.device = config['device']
        # 模型初始化
        self.model = torch.load(model_weight_path, map_location=torch.device(self.device)).to(self.device)
        self.model.eval()
        self.predict_img_conf('./images/welcome.jpg', self.model, self.device)
        self.to_predict_name = "images/logo.png"
        # todo 修改类名，这个数组在模型训练的开始会输出
        self.resize(1200, 800)
        self.initUI()
        self.reset_vid()

    # 界面初始化，设置界面布局
    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        # 主页面，设置组件并在组件放在布局上
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        label_result = QLabel(' 识别结果 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        # 关于页面，设置组件并把组件放在布局上
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('基于深度学习的花朵图像分类算法研究')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/welcome.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        # label_super = QLabel("")
        # label_super.setFont(QFont('楷体', 12))
        # label_super.setOpenExternalLinks(True)
        # label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        # about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        # todo 视频识别界面
        # 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        # vid_title = QLabel("视频检测功能")
        # vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/vid.jpg"))
        # vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)

        self.result2 = QLabel("等待识别")
        self.result2.setFont(QFont('楷体', 24))

        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font)
        self.mp4_detection_btn.setFont(font)
        self.vid_stop_btn.setFont(font)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        # 添加组件到布局上
        # vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)

        vid_detection_layout.addWidget(self.result2, 0, Qt.AlignCenter)

        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # 添加注释
        self.addTab(about_widget, '欢迎界面')
        self.addTab(main_widget, '主界面')
        self.addTab(vid_detection_widget, '视频检测')
        self.setTabIcon(0, QIcon('images/index.png'))
        self.setTabIcon(1, QIcon('images/introduce.png'))

    # 上传并显示图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # 打开文件选择框选择文件
        img_name = openfile_name[0]  # 获取图片名称
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]  # 将图片移动到当前目录
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)  # 打开图片
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # 将图片的大小统一调整到400的高，方便界面显示
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))  # 将图片大小调整到224*224用于模型推理
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))
            self.result.setText("等待识别")

    # 预测图片
    def predict_img(self, img_path='images/target.png', video=False):
        if not video:
            img_path = 'images/target.png'
        predict_cla, conf = self.predict_img_conf(img_path, self.model, self.device)
        preds = predict_cla
        real_confidence = conf
        if real_confidence < 0:
            finalResult = '无关图片'
        else:
            myExcel = openpyxl.load_workbook('./message.xlsx')  # 获取表格文件
            mySheet = myExcel['Sheet1']  # 获取指定的sheet
            finalResult = (mySheet.cell(row=preds + 2, column=2)).value
        if not video:
            self.result.setText(finalResult)  # 在界面上做显示
        else:
            # print(finalResult)
            self.result2.setText(finalResult)

    def predict_img_conf(self, img_path, model, device):
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

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''

    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = 0

        self.result2.setText("等待识别")

        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
        ### 开启视频文件检测事件 ### 
        '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
       ### 视频开启事件 ### 
    '''

    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def detect_vid(self):
        # 打开视频文件
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu还是cpu进行识别
        # plate_rec_model = init_model(device, 'weights/plate_rec_color.pth', is_color=self.is_color)
        # detect_model = attempt_load('weights/plate_detect.pt', map_location=device)  # load FP32 model
        cap = cv2.VideoCapture(self.vid_source)
        # 循环便利每一帧
        while cap.isOpened():
            # 读取一帧
            ret, frame = cap.read()
            if ret:
                # 处理这一帧，例如保存到本地
                cv2.imwrite("images/upload_show_result.jpg", frame)
                pic_function("images/upload_show_result.jpg", "images/upload_show_result.jpg",
                             640, 640)
                self.predict_img(img_path='images/upload_show_result.jpg', video=True)
                self.vid_img.setPixmap(QPixmap("images/upload_show_result.jpg"))
            else:
                # 读取完毕，退出循环
                break
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                self.reset_vid()
                break
        # 释放资源
        self.reset_vid()
        cap.release()

    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()

    '''
    ### 界面重置事件 ### 
    '''

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/vid.jpg"))
        self.vid_source = 0

        self.result2.setText("等待识别")

        self.webcam = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
