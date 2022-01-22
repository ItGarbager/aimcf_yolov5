import time

from pynput.mouse import Listener, Controller, Button, Events

mouse = Controller()


def on_click(x, y, button, pressed):
    oper = '按下' if pressed else '松开'
    print(f'在{(x, y)}地方{oper}')
    if not pressed:
        return False


while True:
    print(1)
    time.sleep(1)
    mouse.move(20, -20)

    # mouse.press(Button.left)
    # mouse.release(Button.left)

    # # Double click
    # mouse.click(Button.left, 1)
    #
    # # scroll two steps down
    # mouse.scroll(0, 500)