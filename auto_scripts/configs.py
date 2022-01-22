import sys

from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
_desktop = QApplication.desktop()
# 获取显示器分辨率大小
_screen_rect = _desktop.screenGeometry()
SCREEN_WIDTH = _screen_rect.width()
SCREEN_HEIGHT = _screen_rect.height()

# 实时显示窗口名称
SCREEN_NAME = 'csgo_detect'
# 游戏内分辨率大小
GAME_X, GAME_Y = (1920, 1080)
# GAME_X, GAME_Y = (SCREEN_WIDTH, SCREEN_HEIGHT)

# mss 截图指定区域
MONITOR = {"top": 0, "left": 0, "width": GAME_X, "height": GAME_Y}


# 重设窗口大小
RESIZE_X = SCREEN_WIDTH // 4
RESIZE_Y = SCREEN_HEIGHT // 4

# 模型文件
WEIGHTS = r'E:\Project\private\DeepLearn\PyTorchLearn\yolov5_6.0\auto_scripts\weights\cf.pt'

# 预测转换图片大小
IMGSZ = (640, 640)

# 置信度
CONF_THRES = .25

# IOU
IOU_THRES = .45

# 方框宽度
LINE_THICKNESS = 4

# 是否显示图像
SHOW_IMG = False

# 是否显示 label
SHOW_LABEL = False
