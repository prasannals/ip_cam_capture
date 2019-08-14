from ip_cam_capture import IpCamAppCapture

ip = '192.168.1.171'
port = 8080

cap = IpCamAppCapture(ip, port)
cap.run()