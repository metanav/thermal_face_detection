import time
import os
import io
import subprocess
import logging
import MLX90640 as mlx90640
from threading import Thread, Condition

class ThermalCamera:
    def __init__(self, fps, inference_callback):
        self.fps          = fps
        self.img_ori      = None
        self.img_out      = None
        self.max_tem      = None
        self.condition    = Condition()
        self.inference_callback = inference_callback
        mlx90640.setup(self.fps)
        
    def capture(self):
        logging.warning("Started Capture")
        while True:
            with self.condition:
                frame = mlx90640.get_frame()
                self.img_ori, self.img_out, self.max_tem = self.inference_callback(frame)
                self.condition.notify_all()

            time.sleep(1.0 / self.fps)

    def start_recording(self):
        thread = Thread(target=self.capture)
        thread.start()
    
    def stop_recording(self):
        mlx90640.cleanup()

if __name__ ==  '__main__':
    def cb(frame):
        print('Frame = {}'.format(frame))

    fps          = 16

    camera = ThermalCamera(fps, cb)
    camera.start_recording()

