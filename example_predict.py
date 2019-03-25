from ip_cam_capture import IpCamPredict

ip = '192.168.1.171'
port = 8080
path = '/home/prasannals/19fastai/course-v3/nbs/dl1/data/handsup_detection/'
cam = IpCamPredict(ip, port, path)
cam.run()