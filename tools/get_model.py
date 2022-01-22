import torch

from models.experimental import attempt_load
from tools.configs import WEIGHTS, IMGSZ
from utils.torch_utils import select_device


def load_model_infos():
    # 获取设备类型
    device = select_device('')

    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(WEIGHTS, map_location=device)
    print('模型加载完毕 ...')
    if half:
        model.half()
        model(torch.zeros(1, 3, *IMGSZ).to(device).type_as(next(model.parameters())))
    return model, device, half
