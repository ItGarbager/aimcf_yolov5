from win32con import SPI_GETMOUSE, SPI_SETMOUSE, SPI_GETMOUSESPEED, SPI_SETMOUSESPEED
from sys import exit, executable
from platform import release
from mouse import mouse_xy
from ctypes import windll
from os import system
from math import atan

import nvidiasmi
import pywintypes
import win32gui
import pynvml


# 预加载为睡眠函数做准备
TimeBeginPeriod = windll.winmm.timeBeginPeriod
HPSleep = windll.kernel32.Sleep
TimeEndPeriod = windll.winmm.timeEndPeriod


# 简单检查gpu是否够格
def check_gpu():
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认卡1
        gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        pynvml.nvmlShutdown()
    except FileNotFoundError as e:
        # pynvml.nvml.NVML_ERROR_LIBRARY_NOT_FOUND
        print(str(e))
        nvidia_smi.nvmlInit()
        gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # 默认卡1
        gpu_name = nvidia_smi.nvmlDeviceGetName(gpu_handle)
        memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
        nvidia_smi.nvmlShutdown()
    if b'RTX' in gpu_name:
        return 2
    memory_total = memory_info.total / 1024 / 1024
    if memory_total > 3000:
        return 1
    return 0


# 高DPI感知
def set_dpi():
    if int(release()) >= 7:
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except AttributeError:
            windll.user32.SetProcessDPIAware()
    else:
        exit(0)


# 检测是否全屏
def is_full_screen(hWnd):
    try:
        full_screen_rect = (0, 0, windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1))
        window_rect = win32gui.GetWindowRect(hWnd)
        return window_rect == full_screen_rect
    except pywintypes.error as e:
        print('全屏检测错误\n' + str(e))
        return False


# 检查是否为管理员权限
def is_admin():
    try:
        return windll.shell32.IsUserAnAdmin()
    except OSError as err:
        print('OS error: {0}'.format(err))
        return False


# 重启脚本
def restart(file_path):
    windll.shell32.ShellExecuteW(None, 'runas', executable, file_path, None, 1)
    exit(0)


# 清空命令指示符输出
def clear():
    _ = system('cls')


# 确认窗口句柄与类名
def get_window_info():
    supported_games = 'Valve001 CrossFire LaunchUnrealUWindowsClient LaunchCombatUWindowsClient UnrealWindow UnityWndClass'
    test_window = 'Notepad3 PX_WINDOW_CLASS Notepad Notepad++'
    emulator_window = 'BS2CHINAUI Qt5154QWindowOwnDCIcon LSPlayerMainFrame TXGuiFoundation Qt5QWindowIcon LDPlayerMainFrame'
    class_name, hwnd_var = None, None
    testing_purpose = False
    while not hwnd_var:  # 等待游戏窗口出现
        millisleep(3000)
        try:
            hwnd_active = win32gui.GetForegroundWindow()
            class_name = win32gui.GetClassName(hwnd_active)
            if class_name not in (supported_games + test_window + emulator_window):
                print('请使支持的游戏/程序窗口成为活动窗口...')
                continue
            else:
                outer_hwnd = hwnd_var = win32gui.FindWindow(class_name, None)
                if class_name in emulator_window:
                    hwnd_var = win32gui.FindWindowEx(hwnd_var, None, None, None)
                elif class_name in test_window:
                    testing_purpose = True
                print('已找到窗口')
        except pywintypes.error:
            print('您可能正使用沙盒,目前不支持沙盒使用')
            exit(0)

    return class_name, hwnd_var, outer_hwnd, testing_purpose


# 比起python自带sleep稍微精准的睡眠
def millisleep(num):
    TimeBeginPeriod(1)
    HPSleep(int(num))  # 减少报错
    TimeEndPeriod(1)


# 移动鼠标
def move_mouse(a, b):
    enhanced_holdback = win32gui.SystemParametersInfo(SPI_GETMOUSE)
    if enhanced_holdback[1]:
        win32gui.SystemParametersInfo(SPI_SETMOUSE, [0, 0, 0], 0)
    mouse_speed = win32gui.SystemParametersInfo(SPI_GETMOUSESPEED)
    if mouse_speed != 10:
        win32gui.SystemParametersInfo(SPI_SETMOUSESPEED, 10, 0)

    mouse_xy(round(a), round(b))

    if enhanced_holdback[1]:
        win32gui.SystemParametersInfo(SPI_SETMOUSE, enhanced_holdback, 0)
    if mouse_speed != 10:
        win32gui.SystemParametersInfo(SPI_SETMOUSESPEED, mouse_speed, 0)


# 简易FOV计算
def FOV(target_move, base_len):
    actual_move = atan(target_move/base_len) * base_len  # 弧长
    return actual_move


# 用户选择
def use_choice(rangemin, rangemax, askstring):
    selection = -1
    while not (rangemax >= selection >= rangemin):
        user_choice = input(askstring)
        try:
            selection = int(user_choice)
            if not (rangemax >= selection >= rangemin):
                print('请在给定范围选择')
        except ValueError:
            print('呵呵...请重新输入')

    return selection
