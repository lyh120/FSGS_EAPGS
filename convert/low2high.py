import os

import cv2
import keras
import numpy as np
from huggingface_hub import from_pretrained_keras
from PIL import Image
from tqdm import tqdm

model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

source_directory = '/home/ubuntu/datasets/Chocolate/images_low'


destination_dir = '/home/ubuntu/datasets/Chocolate/images_enh'

images_list = os.listdir(source_directory)
print(len(images_list))

for original_img in tqdm(images_list):
    image_name = original_img
    original_img = os.path.join(source_directory,original_img)
    low_light_original_img = Image.open(original_img).convert('RGB')
    # print("1")

    image = keras.utils.original_img_to_array(low_light_original_img)
    image = image.astype('float32') / 255.0
    # print("2")


    image = np.expand_dims(image, axis = 0)
    # print("3")

    result = model.predict(image)
    # print("4")
    final_image = result[0] * 255.0
    final_image = final_image.clip(0,255)
    final_image = final_image.reshape((np.shape(final_image)[0],np.shape(final_image)[1],3))
    final_image = np.uint32(final_image)
    arr = Image.fromarray(final_image.astype('uint8'),'RGB')

    arr.save(os.path.join(destination_dir,image_name))