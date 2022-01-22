from win32con import SRCCOPY
from ctypes import windll
import numpy as np
import pywintypes
import win32gui
import win32ui
import mss


class WindowCapture:  # 截图类
    # 类属性
    hwnd, outerhwnd = None, None  # 窗口句柄
    windows_class = None  # 窗口类名
    ratio_h2H, ratio_w2h = 1, 1  # 截图比例(高对于窗口高,宽对于高)
    total_w, total_h = 0, 0  # 窗口内宽高
    cut_w, cut_h = 0, 0  # 截取宽高
    offset_x, offset_y = 0, 0  # 窗口内偏移x,y
    actual_x, actual_y = 0, 0  # 截图左上角屏幕位置x,y
    left_corner = [0, 0]  # 窗口左上角屏幕位置
    errors = 0  # 仅仅显示一次错误
    sct = mss.mss()  # mss截图初始化
    wDC, dcObj, cDC = None, None, None

    # 构造函数
    def __init__(self, window_class, window_hwnd, h2H = 4/9, w2h = 1.6):
        self.windows_class = window_class
        self.ratio_h2H, self.ratio_w2h = h2H, w2h
        self.hwnd = window_hwnd
        try:
            self.outerhwnd = win32gui.FindWindow(window_class, None)
        except pywintypes.error as e:
            print('找窗口错误\n' + str(e))
        if not self.hwnd:
            raise Exception(f'窗口类名未找到: {window_class}')
        self.update_window_info()
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()

    def update_window_info(self):
        try:
            # 获取窗口数据
            window_rect = win32gui.GetWindowRect(self.hwnd)
            client_rect = win32gui.GetClientRect(self.hwnd)
            self.left_corner = win32gui.ClientToScreen(self.hwnd, (0, 0))

            # 确认截图相关数据
            self.total_w = client_rect[2] - client_rect[0]
            self.total_h = client_rect[3] - client_rect[1]
            self.cut_h = int(self.total_h * self.ratio_h2H)
            self.cut_w = int(self.cut_h * self.ratio_w2h)

            if (self.cut_w > min(self.total_w, windll.user32.GetSystemMetrics(0))) or (self.cut_h > min(self.total_h, windll.user32.GetSystemMetrics(1))):
                raise Exception(f'这宽高不行: {self.cut_w} {self.cut_h}')

            self.offset_x = (self.total_w - self.cut_w) // 2 + self.left_corner[0] - window_rect[0]
            self.offset_y = (self.total_h - self.cut_h) // 2 + self.left_corner[1] - window_rect[1]
            self.actual_x = window_rect[0] + self.offset_x
            self.actual_y = window_rect[1] + self.offset_y
        except pywintypes.error as e:
            if self.errors < 2:
                print('获取窗口数据错误\n' + str(e))
                self.errors += 1
            pass

    def get_screenshot(self):  # 只能在windows上使用,画面无法被遮蔽
        self.update_window_info()
        try:
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(self.dcObj, self.cut_w, self.cut_h)
            self.cDC.SelectObject(dataBitMap)
            self.cDC.BitBlt((0, 0), (self.cut_w, self.cut_h), self.dcObj, (self.offset_x, self.offset_y), SRCCOPY)

            # 转换使得opencv可读
            signedIntsArray = dataBitMap.GetBitmapBits(True)
            cut_img = np.frombuffer(signedIntsArray, dtype='uint8')
            cut_img.shape = (self.cut_h, self.cut_w, 4)
            cut_img = cut_img[..., :3]  # 去除alpha
            cut_img = np.ascontiguousarray(cut_img)  # 转换减少错误

            win32gui.DeleteObject(dataBitMap.GetHandle())  # 释放资源
            return cut_img
        except (pywintypes.error, win32ui.error, ValueError) as e:
            print('截图出错\n' + str(e))
            return None

    def get_window_info(self):
        return self.total_w, self.total_h

    def get_cut_info(self):
        return self.cut_w, self.cut_h

    def get_actual_xy(self):
        return self.actual_x, self.actual_y

    def get_window_left(self):
        return win32gui.GetWindowRect(self.outerhwnd)[0]

    def get_side_len(self):
        return int(self.total_h * (2/3))

    def get_region(self):
        self.update_window_info()
        return (self.actual_x, self.actual_y, self.actual_x + self.cut_w, self.actual_y + self.cut_h)

    def grab_screenshot(self):
        scrnshot = np.array(self.sct.grab(self.get_region()), dtype=np.uint8)[..., :3]
        return np.ascontiguousarray(scrnshot)

    def release_resource(self):
        self.wDC, self.dcObj, self.cDC = None, None, None
