import time

from tools.mouse import mouse_xy, mouse_down, mouse_up


def mouse_click_onece(t: float):
    mouse_down()
    time.sleep(t)
    mouse_up()


def mouse_lock_def(aims, mouse, x, y):
    mouse_pos_x, mouse_pos_y = mouse.position
    dist_list = []
    for _, x_c, y_c, _, _ in aims:
        dist = (x * x_c - mouse_pos_x) ** 2 + (y * y_c - mouse_pos_y) ** 2
        dist_list.append(dist)

    # 获取当前离鼠标最近的
    det = aims[dist_list.index(min(dist_list))]
    tag, x_center, y_center, width, height = det
    x_center = x * x_center
    y_center = y * y_center
    # width = x * width
    height = y * height

    coef = 0.971  # 移动系数
    if tag == 1 or tag == 3:
        offset_x = x_center - mouse_pos_x
        offset_y = y_center - mouse_pos_y
        offset_x *= coef

        mouse_xy(offset_x, offset_y)
        mouse_click_onece(0.05)
    elif tag == 0 or tag == 2:
        offset_x = x_center - mouse_pos_x
        offset_y = y_center - height * 2 / 5 - mouse_pos_y
        offset_x *= coef
        mouse_xy(offset_x, offset_y)
        mouse_click_onece(0.05)
