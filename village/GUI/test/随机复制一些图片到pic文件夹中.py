import os
import random
import shutil


def copy_to_pic(data_folder, new_folder, bili):
    # 遍历data文件夹下的所有文件夹
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            # 获取文件夹内所有图片的路径
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            # 计算需要复制的图片数量
            num_images_to_copy = max(1, len(image_files) * bili // 100)
            # 从文件夹中随机选择要复制的图片
            images_to_copy = random.sample(image_files, num_images_to_copy)
            # 将选定的图片复制到new文件夹中
            for image in images_to_copy:
                src = os.path.join(folder_path, image)
                dst = os.path.join(new_folder, image)
                shutil.copyfile(src, dst)


if __name__ == '__main__':
    data_folder = '../../DataSet/final/valid'
    new_folder = './pic'
    copy_to_pic(data_folder, new_folder, 10)
