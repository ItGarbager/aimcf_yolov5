import win32gui
import win32api
import pyautogui

# from pymouse import PyMouse
hwnd_title = {}


def get_all_hwnd(hwnd, mouse):
    if (win32gui.IsWindow(hwnd) and
            win32gui.IsWindowEnabled(hwnd) and
            win32gui.IsWindowVisible(hwnd)):
        hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})


win32gui.EnumWindows(get_all_hwnd, 0)

# m = PyMouse()

for h, t in hwnd_title.items():
    if t:
        print(h, t)
        # if t == 'Counter-Strike: Global Offensive - Direct3D 9':
        #     left, top, right, bottom = win32gui.GetWindowRect(h)
        #     print(left, top, right, bottom)
        #     pyautogui.click(right - 206, bottom - 31)
