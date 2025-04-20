import os


def count_images_in_folder(folder_path):
    """
    统计文件夹中的图片数量
    """
    image_count = sum(1 for filename in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, filename)))
    return image_count


def print_folder_info(main_folder_path):
    """
    打印子文件夹的名称和对应的图片数量，并统计所有图片的总数量
    """
    total_image_count = 0
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(folder_path):
            image_count = count_images_in_folder(folder_path)
            total_image_count += image_count
            print(f"文件夹 {folder_name} : {image_count}")
    print(f"所有图片总数量: {total_image_count}")


if __name__ == "__main__":
    folder_path = './ori'
    # folder_path = './final/train_Aug'
    print_folder_info(folder_path)
    list = []
    for c in os.listdir(folder_path):
        print("  - '" + c + "'")
        # print(c)
        list.append(c)
    print(list)


