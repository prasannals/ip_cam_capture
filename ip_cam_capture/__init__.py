import requests
import numpy as np
import cv2
import os
from time import time

class IpCamCapture():
    def __init__(self, ip, port_number, out_dir = './IpCamCapture/', window_title = 'IpCamCapture', quit_key='q', f_fmt='.jpg'):
        '''
        ip :: string - ip address of the android phone on which Ip Camera app is running
        port_number :: int - port number used by Ip Camera app on the phone
        '''
        if ip is None or port_number is None or ip.strip() == '' or port_number < 0:
            raise AttributeError('Ip cannot be none or empty. port number cannot be None or less than 0.')
        self.shot_url = f'http://{ip}:{str(port_number)}/shot.jpg'
        self.WINDOW_TITLE = window_title
        self.quit_key = quit_key if len(quit_key) == 1 else 'q'
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.f_fmt = f_fmt

    def get_img_from_url(self):
        response = requests.get(self.shot_url)
        img = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img, -1)
        return img

    def run(self):
        while True:
            img = self.get_img_from_url()
            cv2.imshow(self.WINDOW_TITLE, img)
            
            keyPressed = chr(cv2.waitKey(2) & 0xFF)
            print(ord(keyPressed))
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

