import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
import math
def bbox_center(coords: list[tuple[float, float]]):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return (min(xs) + (max(xs) - min(xs)), min(ys) + (max(ys) - min(ys)))
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
img = cv.imread(r'D:\Projects\weld\weldseg\pixel_size\imgs\1b.jpg')
reader = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_blur = cv.GaussianBlur(img_gray,(3,3),5)
clahe = cv.createCLAHE(clipLimit=5)
final_img = clahe.apply(img_blur) + 30
ret, thresh = cv.threshold(final_img,0,1,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# ax.imshow(thresh)
# plt.show()
edges_mid = cv.Canny(image=thresh, threshold1=0, threshold2=1)
lines = cv.HoughLinesP(
    edges_mid, 1, np.pi / 180, 100, minLineLength=900, maxLineGap=25
)

horiz_lines = []
for i in range(lines.shape[0]):
    line = (lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
    plt.plot([line[0],line[2]],[line[1],line[3]])
ax.imshow(thresh)
#plt.plot(cv.cvtColor(img,cv.COLOR_BGR2GRAY))
plt.show()
# line_centers = np.array([(line[2] - line[0], line[1]) for line in horiz_lines])
# bbc = bbox_center(reader.ocr(img)[0][0][0])
# distances = np.linalg.norm(line_centers - bbc, axis=1)
# min_index = np.argmin(distances)

# print(horiz_lines[min_index][2] - horiz_lines[min_index][0])
#clahe = cv.createCLAHE(clipLimit=3)

#img = clahe.apply(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#gray =  cv.GaussianBlur(img,(5,5),5)
# fig = plt.figure(figsize=(6, 4))
# ax = fig.add_subplot()
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# img = np.array(gray)
# thresh_ = int(img[img.shape[0]//2+img.shape[0]//50:img.shape[1]//2+img.shape[1]//50].mean())
# ax.imshow(((img>thresh_)).astype(int))
# plt.show()
# ret, thresh = cv.threshold(img,0,1,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# a,_ = cv.findContours((thresh).astype(np.uint8),cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# areas = []
# for ci in a:
#     areas.append(cv.contourArea(ci))
# #a = a[np.argmax(areas)]
# fig = plt.figure(figsize=(6, 4))
# ax = fig.add_subplot()
# ax.imshow(thresh)#drow conturs
# plt.show()
# print()