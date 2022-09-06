from models.experimental import attempt_load
from utils.general import set_logging
from utils.torch_utils import select_device


def model_load(weights, device='0'):
    # Initialize
    # set_logging()       # 类似记录日志的功能
    device = select_device(device)      # 设置设备

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model，加载FP32模型

    return model
