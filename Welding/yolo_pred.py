import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from pixel_size.utils import get_pixel_real_size
from weld_processing.read_mask import plot_mask_and_point,return_points_and_size
model_path = r'D:\Projects\weld\weldseg\best.pt'
image_path = 'pixel_size/imgs/17a.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

mask= results[0].masks.data[0].cpu().numpy() * 255
plt.plot(mask)
plt.show()
mask = cv2.resize(mask, (W, H))
cv2.imwrite('./output.png', mask)
res = get_pixel_real_size(image_path)
print(res)
top_line_len,top_coords, bot_line_len,bot_coords = return_points_and_size('./output.png')
print(f'top_line_len={top_line_len,top_line_len*res[0]}mm, bot_line_len={bot_line_len,bot_line_len*res[0]}')

plot_mask_and_point('./output.png')
# for result in results:
#     for j, mask in enumerate(result.masks.data):
#         mask = mask.cpu().numpy() * 255
#         mask = cv2.resize(mask, (W, H))
#         cv2.imwrite('./output.png', mask)