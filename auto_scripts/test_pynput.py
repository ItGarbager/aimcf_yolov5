from time import sleep
from pynput.mouse import Controller, Button


def clickAt(Mouse, x, y):
    Mouse.position = (x, y)
    Mouse.press(Button.right)
    Mouse.release(Button.right)


Mouse = Controller()
while True:
    clickAt(Mouse, 100, 100)
# import win32gui, win32ui, win32con, win32api
# import time
#
#
# # 获取当前鼠标的位置
# class Opt:
#     # 当前鼠标位置
#     def cursor_point(self):
#         pos = win32api.GetCursorPos()
#
#         return int(pos[0]), int(pos[1])
#
#     # 移动鼠标位置
#     def mouse_move(self, new_x, new_y):
#         if new_y is not None and new_x is not None:
#             point = (new_x, new_y)
#
#             # print(point)
#             win32api.SetCursorPos(point)
#             self.x = new_x
#             self.y = new_y
#
#     # 鼠标左击事件
#     def mouse_left_click(self, new_x=None, new_y=None, times=1):
#         self.mouse_move(new_x, new_y)
#
#         time.sleep(0.1)
#         while times:
#             win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
#             win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
#             times -= 1
#
#     # 鼠标右击事件
#     def mouse_right_click(self, new_x=None, new_y=None):
#         self.mouse_move(new_x, new_y)
#
#         time.sleep(0.1)
#         win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
#         win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
#
#
# opt = Opt()
# for i in range(1, 100):
#     opt.mouse_move(int(19.2 * i), 100)
#     time.sleep(.5)
