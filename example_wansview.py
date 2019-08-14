from ip_cam_capture import WansViewCapture

ip = '192.168.1.13'
username = 'admin'
password = '123456'

cap = WansViewCapture(username, password, ip)
cap.run()