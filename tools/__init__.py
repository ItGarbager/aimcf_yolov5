import os
from ctypes import windll
from math import atan
from platform import release
from sys import exit, executable
from time import time
from win32con import VK_END
import pywintypes
import win32gui
from pynput.mouse import Button, Listener
from win32api import GetAsyncKeyState

from tools.grabscreen import grab_screen
from tools.mouse import mouse_xy, gmok, msdkok
# 预加载为睡眠函数做准备
from tools.scrnshot import WindowCapture

TimeBeginPeriod = windll.winmm.timeBeginPeriod
HPSleep = windll.kernel32.Sleep
TimeEndPeriod = windll.winmm.timeEndPeriod


# 检测是否全屏
def is_full_screen(hWnd):
    try:
        full_screen_rect = (0, 0, windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1))
        window_rect = win32gui.GetWindowRect(hWnd)
        return window_rect == full_screen_rect
    except pywintypes.error as e:
        print('全屏检测错误\n' + str(e))
        return False


# 比起python自带sleep稍微精准的睡眠
def millisleep(num):
    TimeBeginPeriod(1)
    HPSleep(int(num))  # 减少报错
    TimeEndPeriod(1)


# 移动鼠标
def move_mouse(a, b):
    mouse_xy(round(a), round(b))


# 简易FOV计算
def FOV(target_move, base_len):
    actual_move = atan(target_move / base_len) * base_len  # 弧长
    return actual_move


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


# 高DPI感知
def set_dpi():
    if int(release()) >= 7:
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except AttributeError:
            windll.user32.SetProcessDPIAware()
    else:
        exit(0)


# 加锁换值
def change_withlock(arrays, var, target_var, locker):
    with locker:
        arrays[var] = target_var


# 检测是否存在配置与权重文件
def check_file(file):
    if not os.path.isfile(file):
        print(f'请下载{file}相关文件!!!')
        millisleep(3000)
        exit(0)


# 清空命令指示符输出
def clear():
    _ = os.system('cls')


# 鼠标检测进程
def mouse_detection(array, lock):
    def on_click(x, y, button, pressed):
        if array[0]:
            return False
        change_withlock(array, 1, 1 if pressed and button == Button.left else 0, lock)
        if pressed and button == Button.left:
            print(time() * 1000)
            change_withlock(array, 2, time() * 1000, lock)
        elif not pressed and button == Button.left:
            print(time() * 1000)
            change_withlock(array, 3, time() * 1000, lock)

    with Listener(on_click=on_click) as listener:
        listener.join()  # 阻塞鼠标检测线程


# 转变状态
def check_status(arr, lock):
    if GetAsyncKeyState(VK_END) < 0:  # End
        change_withlock(arr, 14, 1, lock)
    if GetAsyncKeyState(0x31) < 0:  # 1
        change_withlock(arr, 6, 1, lock)
    if GetAsyncKeyState(0x32) < 0:  # 2
        change_withlock(arr, 6, 2, lock)
    if GetAsyncKeyState(0x33) < 0 or GetAsyncKeyState(0x34) < 0:  # 3,4
        change_withlock(arr, 6, 0, lock)
        change_withlock(arr, 18, 0, lock)
    if GetAsyncKeyState(0x46) < 0:  # F恢复移动
        change_withlock(arr, 8, 1, lock)
    if GetAsyncKeyState(0x4A) < 0:  # J停止移动
        change_withlock(arr, 8, 0, lock)
    if GetAsyncKeyState(0x12) < 0:  # Alt恢复开火
        change_withlock(arr, 9, 1, lock)
    if GetAsyncKeyState(0x30) < 0:  # 0停止开火
        change_withlock(arr, 9, 0, lock)
