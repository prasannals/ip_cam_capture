from ip_cam_capture import IpCamCapture

ip = '192.168.1.171'
port = 8080

cap = IpCamCapture(ip, port)
cap.run()