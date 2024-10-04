import rasterio
import time
import tensorflow as tf
import numpy as np
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 34).__str__()
import cv2
from rasterio import features
from rasterio.windows import Window
from rasterio.plot import reshape_as_image, reshape_as_raster

def predict_image_chips(image_path, output_path, chip_size = 512):
    print('entered')
    with rasterio.open(image_path) as src:
        raster = src.read()
        image = reshape_as_image(raster)
    print(image.shape)
    img = np.array(image)
    width = image.shape[1]
    height = image.shape[0]
    print(width, height)
    
    class_colors = {0: [200, 200, 200], 1: [10, 10, 10], 2: [20, 20, 20], 3: [70, 70, 70], 4: [90, 90, 90]}
    class_colors_1 = {0: [200],
                1: [10],
                2: [20],
                3: [70],
                4: [90],
                }
    class_colors_2 = {0: [0, 0, 255],##Mars Red
            1: [122, 202, 245],##Medium Sand
            2: [242, 219, 151],##Lake
            3: [0, 115, 38],##Fir Green
            4: [0, 255, 255],##Solar Yellow
            }

    output_image_categorical = np.empty((height, width), dtype=np.uint8)
    output_image_1_band = np.empty((height, width), dtype=np.uint8)
    output_image_predicted = np.empty((height, width, 3), dtype=np.uint8)
    output_image_final_colors = np.empty((height, width, 3), dtype=np.uint8)
    counter = 0

    print(output_image_categorical.shape)

    for x in range(0, width, chip_size):
        start = time.time()
        for y in range(0, height, chip_size):
            chip_x0 = x  #x * chip_size
            chip_y0 = y  #y * chip_size
            chip_x1 = x + chip_size   #chip_x0 + chip_size
            chip_y1 = y + chip_size #chip_y0 + chip_size
            chip = img[chip_y0:chip_y1, chip_x0:chip_x1]
            print(chip.shape)
            print(x/512,y/512)
            
        
            prediction = model.predict(np.expand_dims(chip, axis=0))
            counter+=1
            print(counter)

            categorical_mask = np.argmax(prediction[0], axis=2)
            cat_mask = categorical_mask.astype(np.uint8)

            output_1band = np.empty((512, 512), dtype = np.uint8)
            for label, color in class_colors_1.items():
                output_1band[categorical_mask == label] = color

            output_image = np.empty((512, 512, 3), dtype = np.uint8)
            for label, color in class_colors.items():
                output_image[categorical_mask == label] = color

            output_final_color = np.empty((512, 512, 3), dtype = np.uint8)
            for label, color in class_colors_2.items():
                output_final_color[categorical_mask == label] = color

            output_image_predicted[chip_y0:chip_y1, chip_x0:chip_x1] = output_image
            output_image_1_band[chip_y0:chip_y1, chip_x0:chip_x1] = output_1band
            output_image_categorical[chip_y0:chip_y1, chip_x0:chip_x1] = cat_mask
            output_image_final_colors[chip_y0:chip_y1, chip_x0:chip_x1] = output_final_color
        print((counter/26864)*100)
        end = time.time()
        print(end-start)
    print(output_image_predicted.shape)
    return output_image_predicted, output_image_1_band, output_image_categorical, output_image_final_colors
        
        

##    with rasterio.open('predicted_mask.tif', 'w', driver = 'GTiff', width=width, height=height, count=2, dtype=np.uint8) as dst:
##        dst.write(output_image_predicted[1, ...])
##
##    with rasterio.open('1band_mask.tif', 'w', driver = 'GTiff', width=width, height=height, count=3, dtype=np.uint8) as dst:
##        dst.write(output_image_1_band[1, ...])
##
##    with rasterio.open('cat_mask.tif', 'w', driver = 'GTiff', width=width, height=height, count=3, dtype=np.uint8) as dst:
##        dst.write(output_image_categorical[1, ...])

##        save_prediction(prediction, x, y)

##def save_prediction(prediction, x, y):
##    with open(f'prediction_{x}_{y}.tif', 'wb') as f:
##        f.write(prediction)

##image_path = r'G:\VIT_Test\INFERENCING\New folder\Clipped_patch.tif'
##image_path = r'G:\VIT_Test\1024\img\img_1024_100.TIF'
image_path = r'G:\VIT_Test\INFERENCING\Total Image\Coconut_Patch.tif'
model_path = r'G:\VIT_Test\512_trials\Deeplab\15 Ep_ 5,10,15_BS16\trial_deeplab_512_15ep_5class_dropout_bs16_dil_5_10_15.h5'
model = tf.keras.models.load_model(model_path)
output_path = r'G:\VIT_Test\INFERENCING\Total Image'
start1 = time.time()
pred_img, band1, cat_mask, final_colors = predict_image_chips(image_path, output_path, 512)
end1 = time.time()
print(end1 - start1)

band_n = '1band_coconut_mask.tif'
cat_msk_img = 'cat_mask.tif'
predicted_image = 'predicted_mask.tif'
final_colors_name = 'final_colors_DeepLab_512_1.tif'
os.chdir(r'G:\VIT_Test\INFERENCING\Total Image')
##cv2.imwrite(predicted_image, pred_img)
##cv2.imwrite(cat_msk_img, cat_mask)
cv2.imwrite(band_n, band1)
##cv2.imwrite(final_colors_name, final_colors)
