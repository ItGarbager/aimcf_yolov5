import time

from tools import FOV, move_mouse
from tools.configs import SHOT_SPEED, MONITOR, DPI
from tools.mouse import mouse_down, mouse_up


# 一次点击
def mouse_click_onece(t: float):
    mouse_down()
    time.sleep(t)
    mouse_up()


# 驱动实现自瞄
def mouse_lock_def(aims, mouse, LOCK_PRESS, C, pids, move_factor, arr, left, top, width, height):
    mouse_pos_x, mouse_pos_y = mouse.position
    # pid
    pid_x, pid_y = pids
    # xy_dist_list = []
    # for tag, x_c, y_c, w, h in aims:
    #     # # 方式一利用中心点算最优
    #     xy_dist = (width * x_c + left - mouse_pos_x) ** 2 + (height * y_c + top - mouse_pos_y) ** 2
    #     xy_dist_list.append(xy_dist)
    #
    # # 获取当前离鼠标最近的
    # det = aims[xy_dist_list.index(min(xy_dist_list))]
    # tag, x, y, _, h = det
    # x = x * width + left
    # y = y * height + top
    # h = h * height

    best_xy = None
    for tag, x, y, _, h in aims:
        x = x * width + left
        y = y * height + top
        h = h * height
        dist = ((x - mouse_pos_x) ** 2 + (y - mouse_pos_y) ** 2) * .5
        if not best_xy:
            best_xy = ((tag, x, y, h), dist)
        else:
            _, old_dist = best_xy
            if dist < old_dist:
                best_xy = ((tag, x, y, h), dist)
    tag, x, y, h = best_xy[0]

    if tag == 1 or tag == 3:  # cf
        move_x = x - mouse_pos_x
        move_y = y - mouse_pos_y
        # mouse_xy(offset_x, offset_y)
        # mouse_click_onece(0.05)
    else:
        move_x = x - mouse_pos_x
        move_y = y - h * 2 / 5 - mouse_pos_y
        # mouse_xy(offset_x, offset_y)
        # mouse_click_onece(0.05)

    # 判断是否需要移动
    if move_x or move_y:
        # move_x = FOV(move_x, x) / DPI * move_factor
        # move_y = FOV(move_y, y) / DPI * move_factor

        move_x = - pid_x(move_x)
        move_y = - pid_y(move_y)

        move_mouse(move_x, move_y)
        if LOCK_PRESS and C < 42:
            move_mouse(0, 1)
            time.sleep(.009)

            C += 1

        # # 下面的是到达一定范围内，不需要直接注释掉即可 可以自动开枪
        # if time.time() * 1000 - arr[0] > SHOT_SPEED and (move_x ** 2 + move_y ** 2) ** .5 <= 1:
        #     mouse_down(1)
        #     mouse_up(1)
