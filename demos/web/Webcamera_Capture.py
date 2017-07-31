import cv2
import time
for i in range(10):
	cam=cv2.VideoCapture(0)
	s,im=cam.read()
	cv2.imwrite("test"+str(i)+".jpg",im)
	cam.release()
	time.sleep(1)

	


