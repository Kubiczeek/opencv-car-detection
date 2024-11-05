import cv2
import numpy as np

# Load the image
image = cv2.imread('Cars_image.jpg')
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml') # Load the car classifier

# Count the number of cars and display two images - one with the bounding box and one without
cars = car_cascade.detectMultiScale(image, 1.1, 1)
print('Number of cars found: {0}'.format(len(cars)))

for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Write number of cars on the image in the top left corner
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'Number of cars: {0}'.format(len(cars)), (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow('Cars', image)
cv2.waitKey(0)
cv2.destroyAllWindows()