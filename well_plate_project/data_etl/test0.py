import cv2


image_file_name = 'bianc_cropped.jpg'

print('check 0')
# assert image_file.is_file()
queryImage = cv2.imread(image_file_name)
print('check 1')
gray_queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
print('check 2')
#fast = cv2.FastFeatureDetector(40) #FAST algorithm for corner detection
fast = cv2.FastFeatureDetector_create() #FAST algorithm for corner detection
print('check 3')

# test = fast.detectAndCompute(queryImage, None)
keypoints_fast  = fast.detect(queryImage, None)

#keypoints_fast  = fast.detectAndCompute(gray_queryImage, None)


print("Done")
