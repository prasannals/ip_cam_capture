import cv2
from ip_cam_capture import OpenCVCapture

cap = cv2.VideoCapture(0)
opencvCap = OpenCVCapture(cap)
opencvCap.run()