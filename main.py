# Connected Component Analysis for Character Segmentation and Keras-OCR for Character Recognition

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
from skimage.util import invert
from operator import itemgetter
from skimage.transform import resize
import keras_ocr
from PIL import Image, ImageOps
import sys

# Read Image
car = imread(sys.argv[1])
car = resize(car, (car.shape[0]*5, car.shape[1]*5), anti_aliasing=True)

# Convert Image to grayscale
gray_img = rgb2gray(car)

# Apply Gaussian Blur to reduce noise
blurred_gray_img = gaussian(gray_img)
plt.figure(figsize=(20,20))
plt.axis("off")
plt.imshow(blurred_gray_img, cmap="gray")

# Binarize Image using Otsu thresholding to separate foreground from background
thresh = threshold_otsu(gray_img)
# print(thresh)
binary = invert(gray_img > thresh)
# plt.figure(figsize=(20,20))
# plt.axis("off")
plt.imshow(binary, cmap="gray")
plt.savefig('binary_image.png')

# Connected Component Analysis of image, with connectivity set to 2
label_image = label(binary, connectivity=2)
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
ax.imshow(binary, cmap="gray")
for region in regionprops(label_image):
  minr, minc, maxr, maxc = region.bbox
  rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
  ax.add_patch(rect)
plt.tight_layout()
plt.show()

# Remove regions which probably don't contain any text and identify and save regions that contain text
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(blurred_gray_img, cmap="gray")
text_like_regions = []
for region in regionprops(label_image):
  minr, minc, maxr, maxc = region.bbox
  w = maxc - minc
  h = maxr - minr
  asr = w/h
  region_area = w*h
  wid,hei = blurred_gray_img.shape
  img_area = wid*hei
  # The aspect ratio is less than 1 to eliminate highly elongated regions. These regions won't contain text.
  # The size of the region should be greater than 15 pixels but smaller than 1/5th of the image
  # dimension to be considered for further processing. These dimensions can be further tuned to 
  # get better results.
  if region_area > 15 and region_area < (0.2 * img_area) and asr < 1 and h > w:
    text_like_regions.append(region)

all_points = []
for region in text_like_regions:
  minr, minc, maxr, maxc = region.bbox
  rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
  ax.add_patch(rect)
  all_points.append([minc, minr, maxc, maxr])
plt.tight_layout()
plt.show()  

# Cropping out characters and numbers
# Sorting by x coordinate to get characters and numbers in order
all_points = sorted(all_points, key = itemgetter(0))
# print(all_points)

numCharacters = len(all_points)

for i in range(len(all_points)):
  x1 = all_points[i][0]
  y1 = all_points[i][1]
  x2 = all_points[i][2]
  y2 = all_points[i][3]
  croppedimg = binary[y1:y2, x1:x2]
  imsave(f'cropped_img_{i}.png', croppedimg)
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.imshow(croppedimg, cmap="gray")
  plt.show()

# Adding border to cropped out image for better recognition
for i in range(0, numCharacters):
  img = Image.open(f'cropped_img_{i}.png')
  img = ImageOps.expand(img,border=170,fill='black')
  img.save(f'cropped_img_{i}.png')

# Making an OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Character Recognition
images = []
for i in range(0, numCharacters):
  images.append(f'cropped_img_{i}.png')
prediction_groups = pipeline.recognize(images)
for i in range(len(prediction_groups)):
  try:
    print(prediction_groups[i][0][0], end=' ')
  except:
    continue
