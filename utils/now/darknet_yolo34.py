from two_class_threat import threat_handling
from math import sqrt, pow
from util import check_gpu
import numpy as np
import win32gui
import cv2


# 分析类
class FrameDetection34:
    # 类属性
    side_width, side_height = 288, 352  # 512, 320  # 输入尺寸
    conf_thd = 0.5  # 置信度阀值
    nms_thd = 0.3  # 非极大值抑制
    win_class_name = None  # 窗口类名
    class_names = None  # 检测类名
    total_classes = 1  # 模型类数量
    CONFIG_FILE, WEIGHT_FILE = ['./'], ['./']
    COLORS = []
    model, net = None, None  # 建立模型, 建立网络
    errors = 0  # 仅仅显示一次错误

    # 构造函数
    def __init__(self, hwnd_value):
        self.win_class_name = win32gui.GetClassName(hwnd_value)
        self.nms_thd = {
            'Valve001': 0.45,
            'CrossFire': 0.45,
        }.get(self.win_class_name, 0.45)

        load_file('yolov4-tiny', self.CONFIG_FILE, self.WEIGHT_FILE)
        self.net = cv2.dnn.readNet(self.CONFIG_FILE[0], self.WEIGHT_FILE[0])  # 读取权重与配置文件
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.side_width, self.side_height), scale=1/255, swapRB=False)
        try:
            with open('classes.txt', 'r') as f:
                self.class_names = [cname.strip() for cname in f.readlines()]
        except FileNotFoundError:
            self.class_names = ['human-head', 'human-body']
        for i in range(len(self.class_names)):
            self.COLORS.append(tuple(np.random.randint(256, size=3).tolist()))

        # 检测并设置在GPU上运行图像识别
        if cv2.cuda.getCudaEnabledDeviceCount():
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            gpu_eval = check_gpu()
            # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # 似乎需要模型支持半精度浮点
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            gpu_message = {
                2: '小伙电脑顶呱呱啊',
                1: '战斗完全木得问题',
            }.get(gpu_eval, '您的显卡配置不够')
            print(gpu_message)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print('您没有可识别的N卡')

    def detect(self, frames, recoil_coty, windoww = 1600):
        try:
            if frames.any():
                frame_height, frame_width = frames.shape[:2]
            frame_height += 0
            frame_width += 0
        except (cv2.error, AttributeError, UnboundLocalError) as e:
            if self.errors < 2:
                print(str(e))
                self.errors += 1
            return 0, 0, 0, 0, 0, frames

        # 检测
        classes, scores, boxes = self.model.detect(frames, self.conf_thd, self.nms_thd)
        threat_list = []

        # 画框
        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = self.class_names[classid[0]] + ': ' + str(round(score[0], 3))
            x, y, w, h = box
            cv2.rectangle(frames, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frames, label, (int(x + w/2 - 4*len(label)), int(y + h/2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if classid == self.total_classes:
                self.total_classes += 1

            # 计算威胁指数(正面画框面积的平方根除以鼠标移动到目标距离)
            h_factor = (0.5 if w >= h or (self.total_classes > 1 and classid == 0) else 0.25)
            dist = sqrt(pow(frame_width / 2 - (x + w / 2), 2) + pow(frame_height / 2 - (y + h * h_factor), 2))
            threat_var = -(pow(w * h, 1/2) / dist * score if dist else 9999)
            if classid == 0:
                threat_var *= 6
            threat_list.append([threat_var, box, classid])

        x0, y0, fire_pos, fire_close, fire_ok, frames = threat_handling(frames, windoww, threat_list, recoil_coty, frame_height, frame_width, self.total_classes)

        return len(threat_list), int(x0), int(y0), fire_pos, fire_close, fire_ok, frames


# 加载配置与权重文件
def load_file(file, config_filename, weight_filename):
    cfg_filename = file + '.cfg'
    weights_filename = file + '.weights'
    config_filename[0] += cfg_filename
    weight_filename[0] += weights_filename
    return
