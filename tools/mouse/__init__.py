from ctypes import CDLL, c_int, c_int64
from os import path

basedir = path.dirname(path.abspath(__file__))
ghubdlldir = path.join(basedir, 'ghub_mouse.dll')
msdkdlldir = path.join(basedir, 'msdk.dll')

# ↓↓↓↓↓↓↓↓↓ 调用ghub/键鼠驱动 ↓↓↓↓↓↓↓↓↓

gm = CDLL(ghubdlldir)
gmok = gm.Agulll()

msdk = CDLL(msdkdlldir)
M_Open = msdk.M_Open
M_Open.argtypes = [c_int]
M_Open.restype = c_int64
msdk_hdl = M_Open(1)
msdkok = 1 if msdk_hdl else 0

if msdkok:
    M_LeftDown = msdk.M_LeftDown
    M_LeftDown.restype = c_int
    M_LeftDown.argtypes = [c_int64]

    M_RightDown = msdk.M_RightDown
    M_RightDown.restype = c_int
    M_RightDown.argtypes = [c_int64]

    M_LeftUp = msdk.M_LeftUp
    M_LeftUp.restype = c_int
    M_LeftUp.argtypes = [c_int64]

    M_RightUp = msdk.M_RightUp
    M_RightUp.restype = c_int
    M_RightUp.argtypes = [c_int64]

    M_MoveR = msdk.M_MoveR
    M_MoveR.restype = c_int
    M_MoveR.argtypes = [c_int64, c_int, c_int]

    M_MouseWheel = msdk.M_MouseWheel
    M_MouseWheel.restype = c_int
    M_MouseWheel.argtypes = [c_int64, c_int]

    M_KeyDown2 = msdk.M_KeyDown2
    M_KeyDown2.restype = c_int
    M_KeyDown2.argtypes = [c_int64, c_int]

    M_KeyUp2 = msdk.M_KeyUp2
    M_KeyUp2.restype = c_int
    M_KeyUp2.argtypes = [c_int64, c_int]

    M_Close = msdk.M_Close
    M_Close.restype = c_int
    M_Close.argtypes = [c_int64]


def mouse_xy(x, y):
    if gmok:
        return gm.Mach_Move(int(x), int(y))
    elif msdkok:
        return M_MoveR(msdk_hdl, int(x), int(y))


def mouse_down(key=1):
    if gmok:
        return gm.Leo_Kick(int(key))
    elif msdkok:
        if key == 1:
            return M_LeftDown(msdk_hdl)
        elif key == 2:
            return M_RightDown(msdk_hdl)


def mouse_up(key=1):
    if gmok:
        return gm.Niman_years()
    elif msdkok:
        if key == 1:
            return M_LeftUp(msdk_hdl)
        elif key == 2:
            return M_RightUp(msdk_hdl)


def scroll(num=1):
    if gmok:
        return gm.Mebiuspin(int(num))
    elif msdkok:
        return M_MouseWheel(msdk_hdl, -int(num))


def mouse_close():
    if gmok:
        return gm.Shwaji()
    elif msdkok:
        return M_Close(msdk_hdl)


def key_down(key=69):
    if msdkok:
        return M_KeyDown2(msdk_hdl, key)


def key_up(key=69):
    if msdkok:
        return M_KeyUp2(msdk_hdl, key)


# ↑↑↑↑↑↑↑↑↑ 调用ghub/键鼠驱动 ↑↑↑↑↑↑↑↑↑

"""
键盘按键和键盘对应代码表：
A <--------> 65 B <--------> 66 C <--------> 67 D <--------> 68
E <--------> 69 F <--------> 70 G <--------> 71 H <--------> 72
I <--------> 73 J <--------> 74 K <--------> 75 L <--------> 76
M <--------> 77 N <--------> 78 O <--------> 79 P <--------> 80
Q <--------> 81 R <--------> 82 S <--------> 83 T <--------> 84
U <--------> 85 V <--------> 86 W <--------> 87 X <--------> 88
Y <--------> 89 Z <--------> 90 0 <--------> 48 1 <--------> 49
2 <--------> 50 3 <--------> 51 4 <--------> 52 5 <--------> 53
6 <--------> 54 7 <--------> 55 8 <--------> 56 9 <--------> 57
数字键盘 1 <--------> 96 数字键盘 2 <--------> 97 数字键盘 3 <--------> 98
数字键盘 4 <--------> 99 数字键盘 5 <--------> 100 数字键盘 6 <--------> 101
数字键盘 7 <--------> 102 数字键盘 8 <--------> 103 数字键盘 9 <--------> 104
数字键盘 0 <--------> 105
乘号 <--------> 106 加号 <--------> 107 Enter <--------> 108 减号 <--------> 109
小数点 <--------> 110 除号 <--------> 111
F1 <--------> 112 F2 <--------> 113 F3 <--------> 114 F4 <--------> 115
F5 <--------> 116 F6 <--------> 117 F7 <--------> 118 F8 <--------> 119
F9 <--------> 120 F10 <--------> 121 F11 <--------> 122 F12 <--------> 123
F13 <--------> 124 F14 <--------> 125 F15 <--------> 126
Backspace <--------> 8
Tab <--------> 9
Clear <--------> 12
Enter <--------> 13
Shift <--------> 16
Control <--------> 17
Alt <--------> 18
Caps Lock <--------> 20
Esc <--------> 27
空格键 <--------> 32
Page Up <--------> 33
Page Down <--------> 34
End <--------> 35
Home <--------> 36
左箭头 <--------> 37
向上箭头 <--------> 38
右箭头 <--------> 39
向下箭头 <--------> 40
Insert <--------> 45
Delete <--------> 46
Help <--------> 47
Num Lock <--------> 144
; : <--------> 186
= + <--------> 187
- _ <--------> 189
/ ? <--------> 191
` ~ <--------> 192
[ { <--------> 219
| <--------> 220
] } <--------> 221
'' ' <--------> 222
"""
