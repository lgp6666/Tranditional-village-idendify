import Model.Model_MobileNet.model_v2 as model_v2
import Model.Model_MobileNet.model_v3 as model_v3


def create_MobileNet(model_name="MobileNet_V3_Small", num_classes=1000):
    if model_name.lower() == 'MobileNet_V3_Small'.lower():
        return model_v3.mobilenet_v3_small(num_classes)
    elif model_name.lower() == 'MobileNet_V3_Large'.lower():
        return model_v3.mobilenet_v3_large(num_classes)
    elif model_name.lower() == 'MobileNet_V2'.lower():
        return model_v2.MobileNetV2(num_classes)
    else:
        print('参数错误')
    return None
