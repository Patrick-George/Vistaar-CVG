import cv2
import numpy as np
from matplotlib import pyplot as plt 

img = cv2.imread("field.jpg",0)
template = cv2.imread("template.jpg",0)
w,h = template.shape[::-1]
res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left,bottom_right,255,2)
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')

plt.show()
