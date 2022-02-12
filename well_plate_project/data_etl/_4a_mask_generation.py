# -*- coding: utf-8 -*-
"""
Created on Nov 15


Mask a weel plate:


https://stackoverflow.com/questions/62944745/how-to-find-the-average-rgb-value-of-a-circle-in-an-image-with-python

https://datacarpentry.org/image-processing/04-drawing/


 * Python program to mask out everything but the wells
 * in a standardized scanned 96-well plate image, without
 * using a file with well center location.

@author: modal
"""


def wellplate_mask(image):
    import numpy as np
    import skimage
    from skimage.viewer import ImageViewer
    import sys

    # read in original image
    image = skimage.io.imread(sys.argv[1])

    # create the mask image
    mask = np.ones(shape=image.shape[0:2], dtype="bool")

    # upper left well coordinates
    x0 = 91
    y0 = 108

    # spaces between wells
    deltaX = 70
    deltaY = 72

    x = x0
    y = y0

    # iterate each row and column
    for row in range(12):
        # reset x to leftmost well in the row
        x = x0
        for col in range(8):

            # ... and drawing a white circle on the mask
            rr, cc = skimage.draw.circle(y, x, radius=16, shape=image.shape[0:2])
            mask[rr, cc] = False
            x += deltaX
        # after one complete row, move to next row
        y += deltaY

    # apply the mask
    image[mask] = 0

    # write the masked image to the specified output file
    skimage.io.imsave(fname=sys.argv[2], arr=image)

    return image

    #We can draw on skimage images with functions such as skimage.draw.rectangle(), skimage.draw.circle(), skimage.draw.line(), and more.


def single_circle_detect(detected_circles, image):
    x, y, r = detected_circles[0].astype(np.int32)
    roi = image[y - r: y + r, x - r: x + r]

    # generate mask
    width, height = roi.shape[:2]
    mask = np.zeros((width, height, 3), roi.dtype)
    cv2.circle(mask, (int(width / 2), int(height / 2)), r, (255, 255, 255), -1)
    dst = cv2.bitwise_and(roi, mask)

    # filter black color and fetch color values
    data = []
    for i in range(3):
        channel = dst[:, :, i]
        indices = np.where(channel != 0)[0]
        color = np.mean(channel[indices])
        data.append(int(color))

    # opencv images are in bgr format
    blue, green, red = data # (110, 74, 49)
    return data

def detect_circle():
    import cv2 
    import numpy as np 

    # Read image. 
    img = cv2.imread('images/placaTeoricaCompleta_result.jpg') 

    # Convert to grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3))


    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1.2, 20, param1 = 50, 
                param2 = 30, minRadius = 30, maxRadius = 50) 

    # Draw circles that are detected. 
    if detected_circles is not None: 

        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 

            # Draw the circumference of the circle. 
            cv2.circle(img, (a, b), r, (0, 255, 0), 2) 

            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
        
        cv2.imshow("Detected Circle", img) 
        cv2.waitKey(0)
    else:
        None
    return 0