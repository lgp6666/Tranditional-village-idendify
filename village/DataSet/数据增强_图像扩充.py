# -*- coding: utf-8 -*-
import random
from PIL import Image, ImageEnhance
import os
from concurrent.futures import ThreadPoolExecutor
import shutil


# 格式化名称
def process_float(number):
    # 将浮点数乘以100
    result = number * 100
    # 只保留整数部分
    integer_part = int(result)
    # 转为字符串
    result_str = str(integer_part)
    # 保留前三位，不足三位在前面补零
    result_str = result_str.zfill(3)[:3]
    return result_str


# 复制文件夹里的所有图片
def copy_images(src_folder, dst_folder):
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 遍历原始文件夹中的所有文件
    for filename in os.listdir(src_folder):
        # 检查文件是否为图片（根据文件扩展名）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)
            # 复制文件到目标文件夹
            shutil.copy2(src_file, dst_file)
            # print(f"已复制: {src_file} 到 {dst_file}")


# 图像增强主程序
def process_image(picture, saveDir, epoch):
    # 打开图片
    img = Image.open(picture)
    original_size = img.size

    # 亮度变换
    brightness_factor = random.uniform(0.4, 0.8) if random.random() < 0.5 else random.uniform(1.2, 1.6)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    # 对比度变换
    contrast_factor = random.uniform(0.4, 0.8) if random.random() < 0.5 else random.uniform(1.2, 1.6)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # 锐化度变换
    sharpness_factor = random.uniform(0.4, 0.8) if random.random() < 0.5 else random.uniform(1.2, 1.6)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)

    # 随机放大或缩小
    scale_factor = random.uniform(0.7, 0.9) if random.random() < 0.5 else random.uniform(1.1, 1.3)
    width, height = img.size
    img = img.resize((int(width * scale_factor), int(height * scale_factor)))

    # 随机旋转
    rotation_angle = random.uniform(10, 350)
    img = img.rotate(rotation_angle)

    # 调整回原始大小
    img = img.resize(original_size)

    # 保存处理后的图片
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    new_name = os.path.basename(picture)[:-4] + '_e_' + str(epoch) + '_b_' + process_float(
        brightness_factor) + '_c_' + process_float(contrast_factor) + '_sh_' + process_float(
        sharpness_factor) + '_sc_' + process_float(scale_factor) + '_r_' + process_float(rotation_angle) + '.jpg'
    img.save(os.path.join(saveDir, new_name))


# 多线程的处理一个文件夹
def process_images_in_parallel(folder_path, saveDir, epoch):
    pictures = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                os.path.isfile(os.path.join(folder_path, f))]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, picture, saveDir, epoch) for picture in pictures]
        for future in futures:
            future.result()


# 处理文件夹
# 修改 f_subfolders 函数
def f_subfolders(root_path, epoch=1):
    for i in range(epoch):
        current_epoch = i + 1
        print(f"\n======== 第 {current_epoch} 轮增强开始 ========")

        output_root = f"{root_path}_Aug"
        os.makedirs(output_root, exist_ok=True)

        # 直接处理根目录（当没有子目录时）
        if not any(os.path.isdir(os.path.join(root_path, f)) for f in os.listdir(root_path)):
            dest_dir = output_root
            os.makedirs(dest_dir, exist_ok=True)

            if current_epoch == 1:
                copy_images(root_path, dest_dir)

            print(f"🔧 处理顶层目录 [{current_epoch}/{epoch}]: {root_path}")
            process_images_in_parallel(root_path, dest_dir, current_epoch)
        else:
            # 原有子目录处理逻辑
            for entry in os.listdir(root_path):
                src_dir = os.path.join(root_path, entry)
                if not os.path.isdir(src_dir):
                    continue
                # ... 原有处理流程

        print(f"======== 第 {current_epoch} 轮增强完成 ========\n")

if __name__ == '__main__':
    # 增强几轮数据集, 每增加一轮图片数量增加一倍。
    epoch = 2

    f_subfolders(r'./ori/linchuan', epoch=epoch)
    f_subfolders(r'./ori/yuanzhou', epoch=epoch)

    print("处理完成")
