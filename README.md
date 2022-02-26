# aimcf_yolov5-v2

<p dir="auto" style="color: #f10505;font-size: 40px">课程代码在aimtools.zip中！！！</p>

<p dir="auto" style="color: #000ec1;font-size: 30px">up 最近工作赶项目，等结束会给大家带新的视频</p>

使用yolov5算法实现cf自瞄

当前版本需要一定的设备支持，目前是使用 Logitech ghub 实现的鼠标移动，如有更好的方法可以交流

入口文件 main.py 已经对原本的代码进行了重构，现在采用 multiprocessing 进行代码运行，所以可能比较吃 cpu

模型文件在 weights/ 可以添加自己训练好的模型

项目整个的配置都在 tools/configs.py 里面修改  

包括检测窗口大小，模型文件等等

自瞄开关目前设置为 左上侧键, 在 configs.py 中可以修改

```python
# main.py
# 点击监听
def on_click(x, y, button, pressed):
    global LOCK_MOUSE
    if pressed and button == exec(f'button.{AIM_BUTTON}'):
        LOCK_MOUSE = not LOCK_MOUSE
        print('LOCK_MOUSE', LOCK_MOUSE)

# configs.py
# 自瞄开关按键
AIM_BUTTON = 'x2'  # 对应按键的关系表可以百度

# 查不到的自己测试
import pynput
# 鼠标点击监听
def on_click(x, y, button, pressed):
    print(button)


# 监听鼠标事件
listener = pynput.mouse.Listener(on_click=on_click)
listener.start()
while True:
    pass

```