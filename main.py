import math
import signal
import threading
import warnings
from collections import deque
from multiprocessing import Pipe, Process, Array, Lock
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
from tools.mouse_controller import mouse_lock_def
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

# 消除警告信息
warnings.filterwarnings('ignore')
# 加载模型
model, device, half = load_model_infos()

# 获取模型其他参
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

# 自瞄开关
LOCK_MOUSE = False
# 压枪开关
LOCK_PRESS = False

# 压枪计数
C = 0
# 鼠标控制
mouse = pynput.mouse.Controller()

# 进程间共享数组
arr = Array('d', 4)

# 初始化进程数组
arr[0] = time() * 1000  # 左键按下时间
arr[1] = time() * 1000  # 左键抬起时间
arr[2] = 0  # pid控制最新时间
arr[3] = 1  # 实时 fps

# 进程锁
process_lock = Lock()


def img_init(p1):
    print('进程 img_init 启动 ...')
    # 启用 mss 截图
    sct = mss.mss()
    while True:
        # 获取指定位置 MONITOR 大小
        img0 = sct.grab(MONITOR)
        img0 = np.array(img0)
        # 将图片转 BGR
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

        # # 获取指定分辨率
        # img0 = grab_screen(region=tuple(MONITOR.values()))

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

        p1.send((img0, aims))


def img_show(c1, p2, arr):
    print('进程 img_show 启动 ...')
    show_up = False
    show_tips = True
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    while True:
        # 展示窗口
        img0, aims = c1.recv()
        p2.send(aims)
        if show_tips:
            print('传输坐标中 ...')
            show_tips = False
        if SHOW_IMG:
            fps = f'{arr[3]:.2f}'
            cv2.putText(img0, fps, (30, 30), FONT, 1.2, (0, 255, 0), LINE_THICKNESS * 2)
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
                p2.send('exit')
                exit('结束 img_show 进程中 ...')


def get_bbox(c2, arr):
    global LOCK_MOUSE, LOCK_PRESS, C
    print('进程 auto_aim 启动 ...')

    # 鼠标点击监听
    def on_click(x, y, button, pressed):
        global LOCK_MOUSE, LOCK_PRESS, C
        if pressed and button == getattr(button, AIM_BUTTON):
            LOCK_MOUSE = not LOCK_MOUSE
            print('LOCK_MOUSE', LOCK_MOUSE)
        if button == Button.left:
            if pressed:
                change_withlock(arr, 0, time() * 1000, process_lock)  # 更新左键按下时间
                LOCK_PRESS = True
            else:
                change_withlock(arr, 1, time() * 1000, process_lock)  # 更新左键抬起时间
                LOCK_PRESS = False
                move_mouse(0, -C)
                C = 0

    # 监听鼠标事件
    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()

    # 初始化一个事件队列
    process_times = deque()

    # 初始化 pid
    pid_x = PID(0.08, 0, 0, setpoint=0, sample_time=0.006, )
    pid_y = PID(0.08, 0, 0, setpoint=0, sample_time=0.006, )
    pid_xy = (pid_x, pid_y)

    # 测试过的几个游戏的移动系数,鼠标灵敏度设置看备注
    # move_factor = {
    #     'CrossFire': 0.971,  # 32
    #     'Valve001': 1.667,  # 2.5
    #     'LaunchCombatUWindowsClient': 1.319,  # 10.0
    #     'LaunchUnrealUWindowsClient': 0.500,  # 20
    # }.get(WINDOW_CLASS_NAME, 1)
    move_factor = 0.971  # 鼠标速度 40
    while True:
        aims = c2.recv()
        # 花费时间
        last_time = arr[2]
        # 实时 fps
        fps = arr[3]

        if len(process_times) > 119:
            process_times.popleft()
        if isinstance(aims, str):
            exit('结束 get_bbox 进程中 ...')
        else:
            if aims and LOCK_MOUSE:
                p = threading.Thread(target=mouse_lock_def, args=(aims, mouse, LOCK_PRESS, C, pid_xy, move_factor, arr), kwargs=MONITOR)
                p.start()
        current_time = time()
        # 耗费时间
        time_used = current_time - last_time

        # 更新时间
        change_withlock(arr, 2, current_time, process_lock)
        process_times.append(time_used)
        median_time = median(process_times)
        pid_x.sample_time = pid_y.sample_time = median_time
        pid_x.Kp = pid_y.Kp = 1 / pow(fps / 2, 1 / 2)
        # 更新 fps
        change_withlock(arr, 3, 1 / median_time if median_time > 0 else 1 / (median_time + SMALL_FLOAT), process_lock)


if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程：
    p1, c1 = Pipe()
    p2, c2 = Pipe()
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

    # 开始线程
    reader1 = Process(target=get_bbox, args=(c2, arr))
    reader2 = Process(target=img_show, args=(c1, p2, arr))
    writer = Process(target=img_init, args=(p1,))
    # 启动子进程 reader，读取:
    reader1.start()
    reader2.start()
    # 启动子进程 writer，写入:
    writer.start()

    # 等待 reader 结束:
    reader1.join()
    reader2.join()
    # 等待 writer 结束:
    writer.terminate()
    writer.join()
    exit('结束 img_init 进程中 ...')
