import cv2

img = cv2.imread('./images/input.jpg', cv2.IMREAD_COLOR)

g,b,r = cv2.split(img)
gbr_img = cv2.merge((g,b,r))
rbr_img = cv2.merge((r,b,r))
bbr_img = cv2.merge((b,b,g))

cv2.imshow('Original', img)
cv2.imshow('GRB', gbr_img)
cv2.imshow('RBR', rbr_img)
cv2.imshow('BBR', bbr_img)

cv2.waitKey()