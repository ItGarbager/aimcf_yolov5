from auto_scripts.mouse.mouse import mouse_xy, mouse_down, mouse_up


def lock(aims, mouse, x, y, logitech=False, model_type='csgo'):
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

    if not logitech:
        if tag == 0 or tag == 2:
            mouse.position = (x_center, y_center)
        elif tag == 1 or tag == 3:
            mouse.position = (x_center, y_center - height / 4)
    else:
        if model_type == 'csgo':
            if tag == 0 or tag == 2:
                offset_x = x_center - mouse_pos_x
                offset_y = y_center - mouse_pos_y
                offset_x *= 1.3
                mouse_xy(offset_x, offset_y)

            elif tag == 1 or tag == 3:
                offset_x = x_center - mouse_pos_x
                offset_y = y_center - height / 6 - mouse_pos_y
                offset_x *= 1.3
                mouse_xy(offset_x, offset_y)
        else:
            coef = 1
            if tag == 1 or tag == 3:
                offset_x = x_center - mouse_pos_x
                offset_y = y_center - mouse_pos_y
                offset_x *= coef
                mouse_xy(offset_x, offset_y)

            elif tag == 0 or tag == 2:
                offset_x = x_center - mouse_pos_x
                offset_y = y_center - height / 6 - mouse_pos_y
                offset_x *= coef
                mouse_xy(offset_x, offset_y)
    mouse_down()
    # mouse_up()
