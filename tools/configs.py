import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from tools import get_window_info

app = QApplication(sys.argv)
_desktop = QApplication.desktop()
# 获取显示器分辨率大小
_screen_rect = _desktop.screenGeometry()
SCREEN_WIDTH = _screen_rect.width()
SCREEN_HEIGHT = _screen_rect.height()

# 实时显示窗口名称
SCREEN_NAME = 'csgo_detect'

# mss 截图指定区域
# MONITOR = {"left": SCREEN_WIDTH // 3, "top": SCREEN_WIDTH // 4, "width": SCREEN_WIDTH // 3, "height": SCREEN_HEIGHT // 2}
MONITOR = {"left": 810, "top": 440, "width": 300, "height": 200}

# 重设窗口大小
RESIZE_X = 150
RESIZE_Y = 100

# 模型文件
WEIGHTS = 'weights/cf.pt'

# 检测源
SOURCE = 'rtmp://localhost:1935/live/demo'

# 预测转换图片大小
IMGSZ = [640, 640]  # 默认

# 置信度
CONF_THRES = .4  # 大于该置信度的目标才会被显示

# IOU
IOU_THRES = .45

# 方框线条粗细
LINE_THICKNESS = 4

# 是否显示图像
SHOW_IMG = True

# 是否显示 label
SHOW_LABEL = False

# label font
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 自瞄开关按键
AIM_BUTTON = 'x2'  # 对应按键的关系表可以百度

# 主武器射击速度
SHOT_SPEED = 169.4

# 初始化一个尽可能小却小得不过分的数
SMALL_FLOAT = np.finfo(np.float64).eps

# # 寻找读取游戏窗口类型并确认截取位置
# WINDOW_CLASS_NAME, WINDOW_HWND_NAME, WINDOW_OUTER_NAME, _ = get_window_info()

# DPI
DPI = 1  # 默认即可
