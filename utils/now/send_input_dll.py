import os.path
import time
from .const import VK_CODE

from ..dll_meta import make_dll_meta

base_dir = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(base_dir, "libsendinput.dll")


class SendInputDll(metaclass=make_dll_meta(DLL)):

    def isSysKey(self, key: int) -> bool:
        pass

    def sendInputKey(self, key_event: int, key_code: int) -> None:
        pass

    def key_down(self, key_code: int):
        pass

    def key_up(self, key_code: int):
        pass

    def key_click(self, key_code: int, interval: int):
        pass

    def sendInputMouse(self, dx: int, dy: int, dwFlags: int):
        pass

    def move_absolute(self, x: int, y: int):
        pass

    def move_relative(self, dx: int, dy: int):
        pass

    def mouse_left_down(self):
        pass

    def mouse_left_up(self):
        pass

    def mouse_right_down(self):
        pass

    def mouse_right_up(self):
        pass


send_input = SendInputDll()

mouse_move_relative = send_input.move_relative


def mouse_left_click(interval: float):
    send_input.mouse_left_down()
    time.sleep(interval)
    send_input.mouse_left_up()


def key_click(key_name: str, interval=0):
    send_input.key_click(VK_CODE[key_name], int(interval * 1000))


if __name__ == '__main__':
    send_input.mouse_right_down()
    send_input.mouse_right_up()
