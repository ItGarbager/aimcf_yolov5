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
def mouse_lock_def(aims, mouse, x, y, pids, move_factor, arr):
    mouse_pos_x, mouse_pos_y = mouse.position
    # pid
    pid_x, pid_y = pids
    xy_dist_list = []
    head_dist_list = []
    body_dist_list = []
    for tag, x_c, y_c, width, height in aims:
        # # 方式一利用中心点算最优
        xy_dist = (x * x_c - mouse_pos_x) ** 2 + (y * y_c - mouse_pos_y) ** 2
        xy_dist_list.append(xy_dist)

        # 方式二 算三维距离
        dist = width * height
        if tag == 1 or tag == 3:
            head_dist_list.append(dist)
            body_dist_list.append(0)
        else:
            head_dist_list.append(0)
            body_dist_list.append(dist)

    # 头最近的索引
    max_head_index = head_dist_list.index(max(head_dist_list))
    # 头最近的
    head_det = aims[max_head_index]
    xy_head_det = xy_dist_list[max_head_index]

    # 身体最近的索引
    max_body_index = body_dist_list.index(max(body_dist_list))
    # 身体最近
    body_det = aims[max_body_index]
    xy_body_det = xy_dist_list[max_body_index]

    # 获取两种结果中的最优解
    det = head_det if xy_head_det <= xy_body_det else body_det

    # # 获取当前离鼠标最近的
    # det = aims[xy_dist_list.index(min(xy_dist_list))]

    tag, x_center, y_center, width, height = det
    x_center = x * x_center + MONITOR.get('left')
    y_center = y * y_center + MONITOR.get('top')
    # width = x * width
    height = y * height
    if tag == 1 or tag == 3:  # cf
        move_x = x_center - mouse_pos_x
        move_y = y_center - mouse_pos_y
        # mouse_xy(offset_x, offset_y)
        # mouse_click_onece(0.05)
    else:
        move_x = x_center - mouse_pos_x
        move_y = y_center - height * 2 / 5 - mouse_pos_y
        # mouse_xy(offset_x, offset_y)
        # mouse_click_onece(0.05)

    # 判断是否需要移动
    if move_x or move_y:
        # move_x = FOV(move_x, x) / DPI * move_factor

        # move_y = FOV(move_y, y) / DPI * move_factor

        pid_move_x = - pid_x(move_x)
        pid_move_y = - pid_y(move_y)

        move_mouse(pid_move_x, pid_move_y)

        # 下面的是到达一定范围内，不需要直接注释掉即可 可以自动开枪
        if time.time() * 1000 - arr[0] > SHOT_SPEED and (pid_move_y ** 2 + pid_move_x ** 2) ** .5 <= 3:
            mouse_down(1)
            mouse_up(1)
