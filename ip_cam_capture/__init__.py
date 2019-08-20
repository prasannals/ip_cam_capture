import requests
import numpy as np
import cv2
import os
from time import time
from fastai.vision import load_learner, open_image
from functools import partial

def get_img_from_url(url):
    response = requests.get(url)
    img = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img, -1)
    return img

class IpCamCapture():
    def __init__(self, get_image, out_dir = './IpCamCapture/', window_title = 'IpCamCapture', quit_key='q', f_fmt='.jpg'):
        self.get_image, self.WINDOW_TITLE = get_image, window_title
        self.out_dir, self.f_fmt = out_dir, f_fmt
        self.quit_key = quit_key if len(quit_key) == 1 else 'q'
        os.makedirs(out_dir, exist_ok=True)

    def run(self):
        while True:
            img = self.get_image()
            cv2.imshow(self.WINDOW_TITLE, img)
            
            keyPressed = chr(cv2.waitKey(2) & 0xFF)
            if ord(keyPressed) != 255 and keyPressed.isalnum():
                if keyPressed == self.quit_key:
                    cv2.destroyAllWindows()
                    break
                else:
                    cls_dir = os.path.join(self.out_dir, keyPressed)
                    os.makedirs(cls_dir, exist_ok=True)
                    fname = os.path.join(cls_dir, self.get_fname() + self.f_fmt)
                    cv2.imwrite(fname, img)

    def get_fname(self):
        return str(time()).replace('.', '_')

def image_from_cv(read):
    # return None if ret == False else return frame
    ret, frame = read()
    return frame if ret else None

class OpenCVCapture(IpCamCapture):
    def __init__(self, capture, out_dir = './IpCamCapture/', window_title = 'IpCamCapture', quit_key='q', f_fmt='.jpg'):
        '''
        capture - instance of OpenCV cv2.VideoCapture 
        '''
        super().__init__(partial(image_from_cv, read=capture.read), out_dir=out_dir, 
            window_title=window_title, quit_key=quit_key,f_fmt=f_fmt)


class WansViewCapture(OpenCVCapture):
    def __init__(self, username, password, ip, out_dir = './IpCamCapture/', window_title = 'IpCamCapture', quit_key='q', f_fmt='.jpg'):
        # TODO - Add parameter validation
        URL = f'rtsp://{username}:{password}@{ip}/live/ch0'
        cap = cv2.VideoCapture(URL)
        super().__init__(cap, out_dir=out_dir, window_title=window_title, 
            quit_key=quit_key,f_fmt=f_fmt)

class ImageLinkCapture(IpCamCapture):
    def __init__(self, image_url, out_dir = './IpCamCapture/', window_title = 'IpCamCapture', quit_key='q', f_fmt='.jpg'):
        super().__init__(partial(get_img_from_url, url=image_url), out_dir=out_dir, window_title=window_title, 
            quit_key=quit_key,f_fmt=f_fmt)

class IpCamAppCapture(ImageLinkCapture):
    def __init__(self, ip, port_number, out_dir = './IpCamCapture/', window_title = 'IpCamCapture', quit_key='q', f_fmt='.jpg'):
        '''
        ip :: string - ip address of the android phone on which Ip Camera app is running
        port_number :: int - port number used by Ip Camera app on the phone
        '''
        if ip is None or port_number is None or ip.strip() == '' or port_number < 0:
            raise AttributeError('Ip cannot be none or empty. port number cannot be None or less than 0.')
        self.shot_url = f'http://{ip}:{str(port_number)}/shot.jpg'
        super().__init__(self.shot_url, out_dir=out_dir, window_title=window_title, 
            quit_key=quit_key,f_fmt=f_fmt)

class IpCamPredict():
    def __init__(self, ip, port_number, path, export_fname='export.pkl', save_img=True, save_dir='./IpCamPredict/', quit_key='q', pred_loc=(20,20),
                font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(0,0,0), line_type=2, window_title='IpCamPredict', f_fmt='.jpg'):
        '''
        path :: string - path to the folder containing the exported pickle file from fast.ai learner.
        '''
        self.save_img, self.path, self.save_dir, self.pred_loc, self.window_title = save_img, path, save_dir, pred_loc, window_title
        self.font, self.font_scale, self.font_color, self.line_type = font, font_scale, font_color, line_type
        self.f_fmt = f_fmt

        self.quit_key = quit_key if len(quit_key) == 1 else 'q'
        self.shot_url = f'http://{ip}:{str(port_number)}/shot.jpg'
        self.learner = load_learner(path, fname=export_fname)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_img_from_url(self):
        response = requests.get(self.shot_url)
        img = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img, -1)
        return img

    def get_fastai_img(self, img):
        tmp_loc = os.path.join(self.save_dir, 'tmp.jpg')
        cv2.imwrite( tmp_loc , img )
        return open_image(tmp_loc)

    def save_pred_img(self, img, label):
        pred_save_dir = os.path.join(self.save_dir, label)
        os.makedirs(pred_save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(pred_save_dir, self.get_fname() + self.f_fmt ), img)
        
    def get_fname(self):
        return str(time()).replace('.', '_')

    def run(self):
        while True:
            img = self.get_img_from_url()

            if (cv2.waitKey(2) & 0xFF) == ord(self.quit_key):
                cv2.destroyAllWindows()
                break
            else:
                f_img = self.get_fastai_img(img)
                pred = self.learner.predict(f_img)
                pred_label = pred[0].obj
                if self.save_img:
                    self.save_pred_img(img, pred_label)
                cv2.putText(img, pred_label, self.pred_loc, self.font, self.font_scale, self.font_color, self.line_type)
                cv2.imshow(self.window_title, img)

