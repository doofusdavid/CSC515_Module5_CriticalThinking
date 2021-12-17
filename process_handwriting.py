import cv2
import numpy as np

img = cv2.imread('./img/IMG_1185.jpeg', 0)

dilation = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((5, 5), np.uint8))

cv2.imshow('Original', img)
cv2.imshow('Dilation', dilation)
cv2.imshow('Closing', closing)
cv2.imshow('Top Hat', tophat)
cv2.imshow('Black Hat', blackhat)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Convert to binary
