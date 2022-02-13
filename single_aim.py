import signal
import threading
import warnings
from collections import deque
from statistics import median
from sys import platform

import mss
import pynput
import torch
import win32con

from simple_pid import PID
from win32api import GetCurrentProcessId, OpenProcess
from win32process import SetPriorityClass, ABOVE_NORMAL_PRIORITY_CLASS

from tools import *
from tools.configs import *
from tools.get_model import load_model_infos
from tools.single_mouse_controller import mouse_lock_def
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

# 消除警告信息
warnings.filterwarnings('ignore')
# 加载模型
model, device, half, stride, names, imgsz = load_model_infos()
# 自瞄开关
LOCK_MOUSE = False

# 鼠标控制
mouse = pynput.mouse.Controller()

# 初始化进程数组
arr = [
    time() * 1000,  # 左键按下时间
    time() * 1000,  # 左键抬起时间
    0,  # pid控制最新时间
    1  # 实时 fps
]


# 鼠标点击监听
def on_click(x, y, button, pressed):
    global LOCK_MOUSE
    if pressed and button == getattr(button, AIM_BUTTON):
        LOCK_MOUSE = not LOCK_MOUSE
        print('LOCK_MOUSE', LOCK_MOUSE)
    if button == Button.left:
        if pressed:
            arr[0] = time() * 1000  # 更新左键按下时间
        else:
            arr[1] = time() * 1000  # 更新左键抬起时间


if __name__ == '__main__':
    try:
        # 检查管理员权限
        if not is_admin():
            restart(__file__)

        set_dpi()  # 设置高DPI不受影响
        os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 设置工作路径
        check_file(WEIGHTS)  # 如果文件不存在则退出
        print(f'罗技驱动加载状态: {gmok}')
        print(f'飞易来/文盒驱动加载状态: {msdkok}')

        # 提升进程优先级
        if platform == 'win32':
            pid = GetCurrentProcessId()
            handle = OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(1)

        print('启动 ...')
        # 启用 mss 截图
        sct = mss.mss()

        # 初始化一个事件队列
        process_times = deque()

        # 初始化 pid
        pid_x = PID(0.15, 0.0, 0.0, setpoint=0, sample_time=0.006, )
        pid_y = PID(0.15, 0.0, 0.0, setpoint=0, sample_time=0.006, )
        pid_xy = (pid_x, pid_y)
        move_factor = 0.971  # 鼠标速度 40

        # 监听鼠标事件
        listener = pynput.mouse.Listener(on_click=on_click)
        listener.start()
        show_up = False
        font = cv2.FONT_HERSHEY_SIMPLEX
        signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

        # 展示图片
        while True:
            # 获取指定位置 MONITOR 大小
            # img0 = sct.grab(MONITOR)
            # img0 = np.array(img0)
            # # 将图片转 BGR
            # img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

            # 获取指定分辨率
            img0 = grab_screen(region=tuple(MONITOR.values()))

            # # 将图片缩小指定大小
            # img0 = cv2.resize(img0, (SCREEN_WIDTH, SCREEN_HEIGHT))

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

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # 获取类别索引
                        c = int(cls)  # integer class
                        # bbox 中的坐标
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # 图片绘制
                        label = SHOW_LABEL and names[c] or None
                        line = (c, *xywh)  # label format
                        aims.append(line)

                        annotator_t = threading.Thread(target=annotator.box_label, args=(xyxy, label), kwargs={'color': colors(c, True)})
                        annotator_t.start()
            # 花费时间
            last_time = arr[2]
            # 实时 fps
            fps = arr[3]
            if SHOW_IMG:
                cv2.putText(img0, f'{fps:.2f}', (50, 50), font, 1.2, (0, 255, 0), LINE_THICKNESS * 2)

                cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
                # 重设窗口大小
                cv2.resizeWindow(SCREEN_NAME, RESIZE_X, RESIZE_Y)
                cv2.imshow(SCREEN_NAME, img0)

                if not show_up:
                    hwnd = win32gui.FindWindow(None, SCREEN_NAME)
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER |
                                          win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)
                    show_up = not show_up
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    cv2.destroyAllWindows()
                    exit('结束 img_show 进程中 ...')
            if len(process_times) > 119:
                process_times.popleft()
            if isinstance(aims, str):
                exit('结束 get_bbox 进程中 ...')
            else:
                if aims and LOCK_MOUSE:
                    p = threading.Thread(target=mouse_lock_def, args=(aims, mouse, MONITOR.get('width'), MONITOR.get('height'), pid_xy, move_factor, arr))
                    p.start()
            current_time = time()
            # 耗费时间
            time_used = current_time - last_time

            # 更新时间
            arr[2] = current_time
            process_times.append(time_used)
            median_time = median(process_times)
            pid_x.sample_time = pid_y.sample_time = median_time
            pid_x.kp = pid_y.kp = 1 / pow(fps / 3, 1 / 3)
            # 更新 fps
            arr[3] = 1 / median_time if median_time > 0 else 1 / (median_time + SMALL_FLOAT)

    except Exception as e:
        print(e)
        while True:
            pass
