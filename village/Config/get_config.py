import yaml


def load_config():
    file_path = r'C:\Users\Lgp666\Desktop\lunw\code\projectcode\CT-valige\Config\config.yaml'
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config




