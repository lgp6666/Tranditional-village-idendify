import Model.Model_EfficientNet.model_eff_V1 as model_eff_V1
import Model.Model_EfficientNet.model_eff_V2 as model_eff_V2


def create_efficientnet(model_name="EfficientNetV2_S", num_classes=1000):
    if model_name.lower() == 'EfficientNet_B0'.lower():
        return model_eff_V1.efficientnet_b0(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNet_B1'.lower():
        return model_eff_V1.efficientnet_b1(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNet_B2'.lower():
        return model_eff_V1.efficientnet_b2(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNet_B3'.lower():
        return model_eff_V1.efficientnet_b3(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNet_B4'.lower():
        return model_eff_V1.efficientnet_b4(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNet_B5'.lower():
        return model_eff_V1.efficientnet_b5(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNet_B6'.lower():
        return model_eff_V1.efficientnet_b6(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNet_B7'.lower():
        return model_eff_V1.efficientnet_b7(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNetV2_S'.lower():
        return model_eff_V2.efficientnetv2_s(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNetV2_M'.lower():
        return model_eff_V2.efficientnetv2_m(num_classes=num_classes)
    elif model_name.lower() == 'EfficientNetV2_L'.lower():
        return model_eff_V2.efficientnetv2_l(num_classes=num_classes)
    else:
        print('参数错误')
        return None

