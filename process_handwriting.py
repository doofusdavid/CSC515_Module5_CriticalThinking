import cv2
import numpy as np


def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


img = cv2.imread('./img/IMG_1185.jpeg')

img = shadow_remove(img)


# dilation = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((5, 5), np.uint8))
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imggray, (5, 5), 0)
# adjust the threshold to minimize the shadow
(thresh, blackAndWhiteImage) = cv2.threshold(
    imggray, 150, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('Original', img)
# cv2.imshow('Dilation', dilation)
# cv2.imshow('Closing', closing)
# cv2.imshow('Top Hat', tophat)
# cv2.imshow('Black Hat', blackhat)
cv2.imshow('Black and White', blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./img/blackAndWhiteImage.jpg", blackAndWhiteImage)

# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                             cv2.THRESH_BINARY, 11, 2)
# (thresh, blackAndWhiteImage) = cv2.threshold(
#     th3, 110, 255, cv2.THRESH_BINARY_INV)

# for i in range(0, 3):
#     eroded = cv2.erode(blackAndWhiteImage.copy(), None, iterations=i + 1)
#     cv2.imshow("Eroded {} times".format(i + 1), eroded)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

# # apply a series of dilations
# for i in range(0, 3):
#     dilated = cv2.dilate(blackAndWhiteImage.copy(), None, iterations=i + 1)
#     cv2.imshow("Dilated {} times".format(i + 1), dilated)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

kernelSizes = [(3, 3), (5, 5), (7, 7)]
# # loop over the kernels sizes
# for kernelSize in kernelSizes:
#     # construct a rectangular kernel from the current size and then
#     # apply an "opening" operation
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
#     opening = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, kernel)
#     opening = cv2.dilate(opening, kernel, iterations=1)
#     cv2.imshow("Opening: ({}, {})".format(
#         kernelSize[0], kernelSize[1]), opening)

#     cv2.waitKey(0)

opening = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_OPEN, (3, 3))
opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, (3, 3))

cv2.imshow("Opening", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# loop over the kernels a final time
for kernelSize in kernelSizes:
    # construct a rectangular kernel and apply a "morphological
    # gradient" operation to the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    gradient = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Gradient: ({}, {})".format(
        kernelSize[0], kernelSize[1]), gradient)
    cv2.waitKey(0)

cv2.imwrite("img/gradient7x7.jpg", gradient)

# cv2.imshow("adaptive", blackAndWhiteImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
dilation = cv2.dilate(blackAndWhiteImage, np.ones(
    (5, 5), np.uint8), iterations=1)
closing = cv2.morphologyEx(
    blackAndWhiteImage, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
tophat = cv2.morphologyEx(
    dilation, cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
blackhat = cv2.morphologyEx(
    blackAndWhiteImage, cv2.MORPH_BLACKHAT, np.ones((5, 5), np.uint8))
cv2.imwrite('img/Dilation.jpg', dilation)
cv2.imwrite('img/Closing.jpg', closing)
cv2.imwrite('img/Top Hat.jpg', tophat)
cv2.imwrite('img/Black Hat.jpg', blackhat)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
