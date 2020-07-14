import tflite_runtime.interpreter as tflite
import numpy as np
import time
import os
import io
import colorsys
import qrcode
import gd3
from threading import Condition
from PIL import Image
from PIL import ImageDraw
from ThermalCamera import ThermalCamera

def decode_boxes(raw_boxes, anchors):
    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(6):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset    ] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def calculate_IoU(first_box, other_boxes):
    a = np.tile(first_box,(other_boxes.shape[0], 1))
    b = other_boxes
    
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + 1e-5)
    return iou

def weighted_non_max_suppression(boxes, scores):
    if len(boxes) == 0: return [], []
    output_boxes = []
    output_scores = []
    # Sort the detections from highest to lowest score.
    remaining = np.argsort(scores)
    
    while len(remaining) > 0:
        box = boxes[remaining[0]]
        score = scores[remaining[0]]
        first_box = box[:4]
        other_boxes = boxes[remaining, :4]
        iou = calculate_IoU(first_box, other_boxes)
        mask = iou > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        weighted_box = np.copy(box)
        weighted_score = score
        if len(overlapping) > 1:
            coordinates = boxes[overlapping]
            _scores = np.expand_dims(scores[overlapping], 1)
            total_score = _scores.sum()
 
            weighted_box   = (coordinates * _scores).sum(axis=0) / total_score
            weighted_score = total_score / len(overlapping)
        output_boxes.append(weighted_box)
        output_scores.append(weighted_score)
        
    return output_boxes, output_scores

def temperature_to_color(val, vmin, vmax):
    color = ((0,0,1), (0,1,1), (0,1,0), (1,1,0), (1,0,0), (1,0,1), (1,1,1))
    fractBetween = 0.0;
    vmin = vmin - 0.5
    vmax = vmax + 0.5 
    vrange = vmax - vmin

    val = (val - vmin) / vrange;
    if np.isnan(val):
        val = 0.0

    if val <= 0:
        idx1 = 0
        idx2 = 0
    elif val >= 1:
        idx1 = len(color) - 1
        idx2 = len(color) - 1
    else:
        val = val * (len(color) -1)
        idx1 = int(val);
        idx2 = idx1 + 1;
        fractBetween = val - idx1;

    r = int((((color[idx2][0] - color[idx1][0]) * fractBetween) + color[idx1][0]) * 255.0)
    g = int((((color[idx2][1] - color[idx1][1]) * fractBetween) + color[idx1][1]) * 255.0)
    b = int((((color[idx2][2] - color[idx1][2]) * fractBetween) + color[idx1][2]) * 255.0)

    return r, g, b

def face_detect(frame):
    if len(frame) == 0:
        print("Zero length frame")
        return frame

    start_ms = time.time()
    vmin = min(frame)
    vmax = max(frame)
    tem  = np.zeros((24, 32))
    img  = Image.new('RGB', (32, 24), 'black')
    
    for y in range(24):
        for x in range(32):
            val = frame[32 * (23-y) + x]
            rgb = temperature_to_color(val, vmin, vmax)
            img.putpixel((x, y), rgb)
            tem[y][x] = val
        
    tem = np.transpose(tem)
    img = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    img_ori = img.crop((0, 0, img.width, img.width))
    img = img_ori.resize((width, height), Image.BICUBIC)
    
    print("inference time = {:.2f} ms".format((time.time() - start_ms) * 1000.0))
    input_data = np.expand_dims(img, axis=0)
    input_data = (np.float32(input_data) - input_mean) / input_std
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_r = interpreter.get_tensor(output_details[0]['index'])
    output_c = interpreter.get_tensor(output_details[1]['index'])
    scores   = np.clip(output_c, a_min=-100.0, a_max=100.0) #clamp
    scores   = 1.0 / (1.0 + np.exp(-scores))  #sigmoid
    scores   = np.squeeze(scores) #remove last dimension
    boxes    = np.squeeze(decode_boxes(output_r, anchors))

    mask1           = scores >= confidence_score_threshold
    filtered_boxes  = boxes[mask1]
    filtered_scores = scores[mask1]
    
    mask2 = [(b[3] - b[1]) * (b[2] - b[0]) < 0.4 for b in filtered_boxes] 
    filtered_boxes  = filtered_boxes[mask2]
    filtered_scores = filtered_scores[mask2]

    output_boxes, output_scores = weighted_non_max_suppression(filtered_boxes, filtered_scores)
    
    frame, max_tem = draw_rectangle(img_ori, tem, output_boxes, output_scores, 5)

    return frame, max_tem

def draw_rectangle(img_out, tem, output_boxes, output_scores, k=5):
    max_face_tem = 0.0

    if len(output_boxes) > 0:
        top_k_indices = np.argsort(output_scores)[-k:][::-1]
        draw     = ImageDraw.Draw(img_out)
        centroids = []
    

        for index in top_k_indices:
            y_min, x_min, y_max, x_max = output_boxes[index][:4]

            bnd = [
                (x_min * img_out.height , y_min * img_out.width - 1.0), 
                (x_max * img_out.height , y_max * img_out.width)
            ]
            draw.rectangle(bnd, outline='white')
            face_tem = tem[int(bnd[0][1]):int(bnd[1][1]), int(bnd[0][0]):int(bnd[1][0])]
            
            if face_tem.shape[0] != 0 and face_tem.shape[1] != 0:
                max_face_tem = np.amax(face_tem)
                #draw.text((bnd[0][0], bnd[0][1]-2), '{:0.2f}'.format(np.amax(face_temp))

    output_buffer = io.BytesIO()
    #img_out = img_out.crop((0, 0, img_out.width, img_out.height ))
    img_out = img_out.resize((img_out.width * 10, img_out.height * 10), Image.BICUBIC)

    if max_face_tem > 34.8:
        qr = qrcode.QRCode(box_size=2)
        qr.add_data('https://gaizine.com')
        qr.make()
        img_qr = qr.make_image()
        pos = (img_out.size[0] - img_qr.size[0], img_out.size[1] - img_qr.size[1])
        img_out.paste(img_qr, pos)

    img_out.save(output_buffer, format="jpeg")
    frame = output_buffer.getvalue()
    return frame, max_face_tem

if __name__ == '__main__':
    model      = 'model/face_detection_front_32.tflite'
    anchors    = np.load('model/anchors.npy')
    x_scale    = 128.0
    y_scale    = 128.0
    w_scale    = 128.0
    h_scale    = 128.0
    input_mean = 127.5
    input_std  = 127.5
    min_suppression_threshold  = 0.2
    confidence_score_threshold = 0.75

    interpreter = tflite.Interpreter(model_path = model)
    interpreter.allocate_tensors()
    
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    height   = input_details[0]['shape'][1]
    width    = input_details[0]['shape'][2]
    fps      = 8

    thermal_camera = ThermalCamera(fps, face_detect)
    print("Started recording")
    thermal_camera.start_recording()
    print("Init GD")
    gd3.init()
    try:
        prev_face_tem = 0.0
        while True:
            with thermal_camera.condition:
                thermal_camera.condition.wait()
                frame    = thermal_camera.frame
                face_tem = thermal_camera.max_tem
                if prev_face_tem > 34.8 and face_tem > 34.8:
                    time.sleep(20)
                else:
                    gd3.load(frame)
                    gd3.display(face_tem)

                prev_face_tem = face_tem
    finally:
        thermal_camera.stop_recording()

