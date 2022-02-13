import torch

from models.experimental import attempt_load
from tools.configs import WEIGHTS, IMGSZ
from utils.general import check_img_size
from utils.torch_utils import select_device


def load_model_infos():
    # 获取设备类型
    device = select_device('')

    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(WEIGHTS, map_location=device)
    print('模型加载完毕 ...')
    if half:
        model.half()
    # 获取模型其他参
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    imgsz = check_img_size(IMGSZ, s=stride)  # check image size
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    return model, device, half, stride, names, imgsz
