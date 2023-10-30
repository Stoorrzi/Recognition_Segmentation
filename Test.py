import cv2 as cv

img = cv.imread("./IMG_1.jpg")
red = color = (0, 0, 255) 
center_coordinates = (1561, 1772) 
cv.circle(img, center_coordinates, 5, red, 2)
cv.circle(img, (1512, 2016), 5, (255, 0, 0), 10)
# cv.circle(img, (1505, 925), 5, red, 2)
# start_point =(1727, 1191)
# end_point = (1369, 1305)
# thickness = 9
# cv.line(img, start_point, end_point, color, thickness) 
# start_point = (1587, 2647)
# end_point = (1577 ,925)
# cv.line(img, start_point, end_point, color, thickness) 


cv.imshow("Image", img)  
cv.waitKey(0)

