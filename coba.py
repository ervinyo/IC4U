import numpy as np
import cv2
e1 = cv2.getTickCount()
x = 5
y=x**2
y=x*x
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print(time)
e1 = cv2.getTickCount()
z = np.uint8([5])
y=z*z
y=np.square(z)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print(time)
