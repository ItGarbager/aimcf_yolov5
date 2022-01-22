import warnings

import cv2
import mss
import numpy as np
import torch
import win32con
import win32gui

# loadConfig
from configs import *
# 创建一个命名窗口
from get_model import load_model_infos

# 消除警告信息
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

warnings.filterwarnings('ignore')
# loadModel
model, device, half = load_model_infos()

# 获取模型其他参
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)

# 重设窗口大小
cv2.resizeWindow(SCREEN_NAME, RESIZE_X, RESIZE_Y)

# 启用 mss 截图
sct = mss.mss()

while True:
    # --- 图像变化 ---
    # 获取指定位置 MONITOR 大小
    img0 = sct.grab(MONITOR)
    img0 = np.array(img0)
    # 将图片转 BGR
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

    # 将图片缩小指定大小
    img0 = cv2.resize(img0, (RESIZE_X, RESIZE_Y))

    # Padded resize
    img = letterbox(img0, IMGSZ, stride=stride)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()

    # 归一化处理
    img = img / 255.
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    pred = model(img, augment=False, visualize=False)[0]
    # NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=1000)
    aims = []
    for i, det in enumerate(pred):
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # 设置方框绘制
        annotator = Annotator(img0, line_width=LINE_THICKNESS, example=str(names))
        if len(det):
            #  Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # 获取类别索引
                c = int(cls)  # integer class
                # bbox 中的坐标
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                line = (c, *xywh)  # label format

                # 图片绘制
                label = SHOW_LABEL and names[c] or None
                annotator.box_label(xyxy, label, color=colors(c, True))

    # --- 展示窗口 ---
    # 展示窗口
    cv2.imshow(SCREEN_NAME, img0)
    hwnd = win32gui.FindWindow(None, SCREEN_NAME)
    CVRECT = cv2.getWindowImageRect(SCREEN_NAME)
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER |
                          win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        cv2.destroyAllWindows()
        exit("Escape hit, closing...")
