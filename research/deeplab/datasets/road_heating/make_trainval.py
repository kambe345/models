import glob
import os
import pandas as pd
import shutil

image_root = './JPEGImages/'
images = glob.glob(os.path.join(image_root, '**/*.jpg'), recursive=True)
for image in images:
    if not os.path.exists(os.path.join(image_root, os.path.basename(image))):
        shutil.move(image, image_root)

image_root = './SegmentationClass/'
images = glob.glob(os.path.join(image_root, '**/*.png'), recursive=True)
for image in images:
    if not os.path.exists(os.path.join(image_root, os.path.basename(image))):
        shutil.move(image, image_root)

images = glob.glob(os.path.join(image_root, '**/*.png'), recursive=True)
images = [os.path.basename(image)[:-4] for image in images]

train_ratio = 0.8
num_images = len(images)
num_train = int(num_images * train_ratio)
num_val = num_images - num_train

images = pd.DataFrame(images)

train_images = images[:num_train]
val_images = images[num_train:]

images.to_csv('./ImageSets/Segmentation/trainval.txt', index=None, header=None)
train_images.to_csv('./ImageSets/Segmentation/train.txt', index=None, header=None)
val_images.to_csv('./ImageSets/Segmentation/val.txt', index=None, header=None)

print('num_images:', num_images)
print('num_train:', num_train)
print('num_val:', num_val)
