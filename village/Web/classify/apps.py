from django.apps import AppConfig
from .torch_model import GlobalObject  # 引入你的单例类


class ClassifyConfig(AppConfig):
    name = 'classify'

    def ready(self):
        # 这里在应用启动时初始化全局对象
        global global_object_instance
        global_object_instance = GlobalObject()