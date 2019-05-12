import os
import cv2


dir = './new_images'
files = os.listdir(dir)

for file in files:
    path = dir+'/'+file
    image = cv2.imread(path, 1)
    img = cv2.imread(path, 0)

    h, w = img.shape
    center = (w//2, h//2)
    for angle in list(range(-10, 11, 1)):
        if angle != 0:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            name = file.replace('.png', '')
            print('%s/%s_%s.png' % (dir, name, str(angle)))
            cv2.imwrite('%s/%s_%s.png' % (dir, name, str(angle)), rotated)
