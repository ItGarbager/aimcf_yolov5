from two_class_threat import threat_handling
from math import sqrt, pow
from util import check_gpu
from numba import njit
import numpy as np
import onnxruntime
import win32gui
import cv2


# 分析类
class FrameDetectionX:
    # 类属性
    std_confidence = 0.5  # 置信度阀值
    nms_thd = 0.3  # 非极大值抑制
    win_class_name = None  # 窗口类名
    class_names = None  # 检测类名
    total_classes = 1  # 模型类数量
    COLORS = []
    WEIGHT_FILE = ['./']
    input_shape = (224, 192)  # 输入尺寸
    EP_list = onnxruntime.get_available_providers()  # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] Tensorrt优先于CUDA优先于CPU执行提供程序
    session, io_binding, device_name = None, None, None
    errors = 0  # 仅仅显示一次错误

    # 构造函数
    def __init__(self, hwnd_value):
        self.win_class_name = win32gui.GetClassName(hwnd_value)
        self.nms_thd = {
            'Valve001': 0.45,
            'CrossFire': 0.45,
        }.get(self.win_class_name, 0.45)
        load_file('yolox_tiny', self.WEIGHT_FILE)

        # 检测是否在GPU上运行图像识别
        self.device_name = onnxruntime.get_device()
        try:
            self.session = onnxruntime.InferenceSession(self.WEIGHT_FILE[0], None)  # 推理构造
        except RuntimeError:
            self.session = onnxruntime.InferenceSession(self.WEIGHT_FILE[0], providers=['CPUExecutionProvider'])  # 推理构造
            # self.session.set_providers('CPUExecutionProvider')
            self.device_name = 'CPU'
        if self.device_name == 'GPU':
            gpu_eval = check_gpu()
            gpu_message = {
                2: '小伙电脑顶呱呱啊',
                1: '战斗完全木得问题',
            }.get(gpu_eval, '您的显卡配置不够')
            print(gpu_message)
        else:
            print('中央处理器烧起来')
            print('PS:注意是否安装CUDA')

        self.io_binding = self.session.io_binding()

        try:
            with open('classes.txt', 'r') as f:
                self.class_names = [cname.strip() for cname in f.readlines()]
        except FileNotFoundError:
            self.class_names = ['human-head', 'human-body']
        for i in range(len(self.class_names)):
            self.COLORS.append(tuple(np.random.randint(256, size=3).tolist()))

    def detect(self, frames, recoil_coty, windoww = 1600):
        try:
            if frames.any():
                frame_height, frame_width = frames.shape[:2]
            frame_height += 0
            frame_width += 0
        except (cv2.error, AttributeError, UnboundLocalError) as e:
            if self.errors < 2:
                print(str(e))
                self.errors += 1
            return 0, 0, 0, 0, 0, frames

        # 预处理
        img, ratio = preprocess(frames, self.input_shape)

        # 检测
        if self.device_name == 'GPU':
            ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(img[None, :, :, :], 'cuda', 0)
            self.io_binding.bind_input(name=self.session.get_inputs()[0].name, device_type=ortvalue.device_name(), device_id=0, element_type=np.float32, shape=ortvalue.shape(), buffer_ptr=ortvalue.data_ptr())
            self.io_binding.bind_output('output')
            self.session.run_with_iobinding(self.io_binding)
            output = self.io_binding.copy_outputs_to_cpu()[0]
        else:
            ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
            output = self.session.run(None, ort_inputs)[0]

        predictions = demo_postprocess(output, self.input_shape)[0]
        boxes_xyxy, scores = analyze(predictions, ratio)
        dets = multiclass_nms(boxes_xyxy, scores, self.nms_thd, self.std_confidence)

        # 画框
        threat_list = []
        if not (dets is None):
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for (box, final_score, final_cls_ind) in zip(final_boxes, final_scores, final_cls_inds):
                cv2.rectangle(frames, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), self.COLORS[0], 2)
                label = str(round(final_score, 3))
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                cv2.putText(frames, label, (int(x + w/2 - 4*len(label)), int(y + h/2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if final_cls_ind == self.total_classes:
                    self.total_classes += 1

                # 计算威胁指数(正面画框面积的平方根除以鼠标移动到目标距离)
                h_factor = (0.5 if w >= h or (self.total_classes > 1 and final_cls_ind == 0) else 0.25)
                dist = sqrt(pow(frame_width / 2 - (x + w / 2), 2) + pow(frame_height / 2 - (y + h * h_factor), 2))
                threat_var = -(pow(w * h, 1/2) / dist * final_score if dist else 9999)
                threat_list.append([threat_var, [x, y, w, h], final_cls_ind])

        x0, y0, fire_pos, fire_close, fire_ok, frames = threat_handling(frames, windoww, threat_list, recoil_coty, frame_height, frame_width, self.total_classes)

        return len(threat_list), int(x0), int(y0), fire_pos, fire_close, fire_ok, frames


# 分析预测数据
@njit(fastmath=True)
def analyze(predictions, ratio):
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio

    return boxes_xyxy, scores


# 从yolox复制的预处理函数
def preprocess(img, input_size, swap=(2, 0, 1)):
    padded_img = np.ones((input_size[0], input_size[1], 3)) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


# 从yolox复制的单类非极大值抑制函数
@njit(fastmath=True)
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


# 从yolox复制的多类非极大值抑制函数
def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


# 从yolox复制的多类非极大值抑制函数(class-agnostic方式)
def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


# 从yolox复制的多类非极大值抑制函数(class-aware方式)
def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


# 从yolox复制的后置处理函数
def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


# 加载配置与权重文件
def load_file(file, weight_filename):
    weights_filename = file + '.onnx'
    weight_filename[0] += weights_filename
    return
