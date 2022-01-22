from cv2 import line


def threat_handling(frame, window_w, threat_alist, recoil_ctrl, frame_height, frame_width, class_num):
    x0, y0, fire_pos, fire_close, fire_ok = 0, 0, 0, 0, 0
    if len(threat_alist):
        threat_alist.sort(key=lambda x:x[0])
        x_tht, y_tht, w_tht, h_tht = threat_alist[0][1]
        fire_close = (1 if window_w / w_tht <= 50 else 0)

        # 指向距离最近威胁中心的位移(class0为头部,class1为身体)
        x0 = x_tht + (w_tht - frame_width) / 2
        y0 = y_tht + (h_tht - frame_height) / 2

        if abs(x0) <= 1/4 * w_tht and abs(y0) <= 2/5 * h_tht:  # 查看是否已经指向目标
            fire_ok = 1

        if threat_alist[0][2] == 0 and class_num > 1:
            fire_pos = 1
        elif h_tht > w_tht:
            y0 -= h_tht / 4
            fire_pos = 2
            if fire_close:
                y0 -= h_tht / 8
                fire_pos = 1

        y0 += recoil_ctrl  # 后座控制
        xpos, ypos = x0 + frame_width / 2, y0 + frame_height / 2
        line(frame, (frame_width // 2, frame_height // 2), (int(xpos), int(ypos)), (0, 0, 255), 2)

    return x0, y0, fire_pos, fire_close, fire_ok, frame
