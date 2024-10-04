import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
###import tensorflow_datasets as tfds
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
###from tensorflow.keras import mixed_precision as mixed_precision
from tensorflow.keras.regularizers import l2
###from tensorflow.python.profiler import profiler_client
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from keras import layers, Model
from pathlib import Path
import shutil

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, add, ZeroPadding2D , UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input , Dropout , GlobalAveragePooling2D , Dense , Multiply # Use ResNet50 instead of ResNet34
#from keras.engine.keras_tensor import KerasTensor

# enable gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=36000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# read user input from text file
fh = open(r"G:\VIT_Test\1024_196_trials\DeepLab\deeplab_inputs_512.txt", 'r')
parameters = {}
for line in fh:
    if line.startswith("\n") or line.startswith("#"):
        continue
    parameter, value = line.split('=')
    parameter = parameter.strip()
    value = value.strip()
    parameters[parameter] = value
print(parameters)


# MANDATORY PARAMETERS
# all parameters that we should be able to change at each run of the program
sample_image_path =  parameters['first_image_path'] #path of first image #MANDATORY
sample_mask_path = parameters['first_mask_path'] #path of first mask #MANDATORY

img_path = os.path.dirname(sample_image_path)
mask_path = os.path.dirname(sample_mask_path)

img_width = int(parameters['img_height'])
img_height = int(parameters['img_width'])
img_channels = int(parameters['number_of_bands_in_image'])
img_shape = (img_width,img_height,img_channels)

class_color_file_location = parameters['class_color_file_location']


#OPTIONAL PARAMETERS

try:
    num_of_samples = int(parameters['number_of_samples'])  #number of samples
except:
    images = len(os.listdir(img_path))
    num_of_samples = images

#model_choice = 'Resnet50' #user choice

try:
    validation_split_size = float(parameters['validation_size'])
except:
    validation_split_size = 0.3

try:
    include_weights = parameters['include_weights']
    if include_weights == 'yes' or include_weights == 'YES':
        try:
            weight_file_location = parameters['weight_file_location']
        except:
            print("Weight file location needed to include weights")
            exit(0)
except:
    include_weights = 'no'


try:
    dropout_percentage = float(parameters['dropout_percentage'])
except:
    dropout_percentage = 0.25 #user choice


try:
    learning_rate = float(parameters['learning_rate'])
except:
    learning_rate = 0.001 #user choice

    
adam_optimiser = tf.keras.optimizers.Adam(learning_rate = learning_rate)
SGD_optimiser = tf.keras.optimizers.SGD(learning_rate = learning_rate)
try:
    optimiser_choice = parameters['optimiser']
except:
    optimiser_choice = 'adam'
if optimiser_choice=='adam':
    optimiser = adam_optimiser
elif optimiser_choice=='SGD':
    optimiser = SGD_optimiser

stride_len = (2,2)


try:
    loss_function = parameters['loss_function']
except:
    loss_function = 'categorical_crossentropy'


metrics = ['accuracy'] #user choice
try:
    batch_size = int(parameters['batch_size'])
except:
    batch_size = 2 #user choice
try:
    num_epochs = int(parameters['num_epochs'])
except:
    num_epochs = 10#user choice
##steps_per_epoch = num_of_samples//batch_size






# making code modular in structure by using functions

def load_dataset(sample_image_path, sample_mask_path, num_of_samples):
    img_dataset = []
    mask_dataset = []
    digit_list = ['0','1','2','3','4','5','6','7','8','9','(',')']

    # Split the image names into usable chunks
    img_path = os.path.dirname(sample_image_path)
    img_name, img_extension = os.path.split(sample_image_path)[-1].split('.')
    
    for i in reversed(range(len(img_name))):
        if img_name[i] not in digit_list:
            img_basename = img_name[:i]
            break
    mask_path = os.path.dirname(sample_mask_path)
    mask_name, mask_extension = os.path.split(sample_mask_path)[-1].split('.')
    for i in reversed(range(len(mask_name))):
        if mask_name[i] not in digit_list:
            mask_basename = mask_name[:i]
            break


    mask_colors = []
    ctr = 0

    # iterate through directory which contains images
    for i in os.listdir(img_path):
        if i.endswith("TIF")==False:
            continue
        if ctr<num_of_samples:

            # construct the image and mask to be read
            img_file_name = i.split('.')[0]
            img_file = img_file_name+'.'+img_extension
            mask_file_name = img_file_name.replace(img_basename, mask_basename)
            mask_file = mask_file_name+'.'+mask_extension

            #read both image and mask and store in np array
            img = cv2.imread(os.path.join(img_path,img_file))
            mask = cv2.imread(os.path.join(mask_path,mask_file))
            
            # failsafe to ignore files which do not exist or do not read or not in proper format
            if img is None or mask is None:
                print("one of both files is not readable, skipping both")
                print(f"{img_file} and {mask_file}")
                continue

            # to find the unique colors in each mask image
            uniq_per_mask = np.unique(mask, axis=0)
            flattened_image_array = uniq_per_mask.reshape(-1, uniq_per_mask.shape[-1])
            unique_colors = set(tuple(color) for color in flattened_image_array) #get unique values by using property of tuple:no duplicate items
            unique_colors = [list(color_tuple) for color_tuple in unique_colors]
            mask_colors.append(unique_colors)

            # create datasets
            img_dataset.append(img)
            mask_dataset.append(mask)

            ctr = ctr+1
    
    img_dataset_arr = np.array(img_dataset)
    del img_dataset
    mask_dataset_arr = np.array(mask_dataset)
    del mask_dataset

    print("image and mask dataset shape")
    print(img_dataset_arr.shape) #(8,1024,1024,3)
    print(mask_dataset_arr.shape) #(8,1024,1024,3)
    return img_dataset_arr, mask_dataset_arr, mask_colors

def no_of_classes(mask_colors):
    #finding number of classes in masks by finding number of unique colours
    unique_colors = []
    for arr in mask_colors:
        for i in arr:
            is_unique = True
            for k in unique_colors:
                if np.array_equal(i,k):
                    is_unique = False
                    break
            if is_unique:
                unique_colors.append(i)

    class_colors = {}
    for i in range(len(unique_colors)):
        class_colors[i] = list(unique_colors[i])
    num_classes = len(class_colors)
    print("classes present in masks")
    print(class_colors)
    return class_colors, num_classes

def normalise_images_for_training(img_dataset_arr, num_of_samples):
    # normalising images for training
    images = []
    for i in range(len(img_dataset_arr)):
        img = img_dataset_arr[i]
        
        pixels_img = img.reshape(-1,img.shape[-1])
        scaled_pixels_img = np.empty_like(pixels_img, dtype=float)
        for band in range(img.shape[-1]):
            scaler = MinMaxScaler()
            scaled_band = scaler.fit_transform(pixels_img[:, band].reshape(-1,1))
            scaled_pixels_img[:, band] = scaled_band.flatten()
            scaled_image = scaled_pixels_img.reshape(img.shape)
        images.append(scaled_image)
    master_images_dataset = np.array(images)
    return master_images_dataset

def one_hot_encode_masks(mask_dataset_arr):
    # one-hot encoding the masks
    def rgb_to_label(label):
        label_segment = np.zeros(label.shape , dtype=np.uint8)
        for i in range(len(class_colors)):
            label_segment[np.all(label == class_colors[i],axis = -1)] = i
        label_segment = label_segment[:,:,0]
        return label_segment

    labels = []
    for i in range(mask_dataset_arr.shape[0]):
        label = rgb_to_label(mask_dataset_arr[i])
        labels.append(label)
    labels = np.expand_dims(labels,axis = 3)
    master_label_dataset = to_categorical(labels,len(np.unique(labels)))
    return master_label_dataset


def split_data_for_train_test(img_dataset_arr, mask_dataset_arr, split_size=0.3):
    # splitting the dataset first into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(img_dataset_arr, mask_dataset_arr, test_size = split_size/2, random_state = 91)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = split_size*0.588, random_state = 91) 
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def split_name_for_train_test(img_dataset_name, mask_dataset_name, split_size=0.3):
    # splitting the dataset first into training and testing
    X_train_name, X_test_name, Y_train_name, Y_test_name = train_test_split(img_dataset_name, mask_dataset_name, test_size = split_size/2, random_state = 91)
    X_train_name, X_val_name, Y_train_name, Y_val_name = train_test_split(X_train_name, Y_train_name, test_size = split_size*0.588, random_state = 91) 
    return X_train_name, X_val_name, X_test_name, Y_train_name, Y_val_name, Y_test_name

##img_dataset_name = list(Path("G:\VIT_Test\1024\img").glob("img_1024_(*).tif"))
####for i in range(len(img_dataset_name)):
##print(img_dataset_name)
##mask_dataset_name = tf.io.gfile.glob("G:\VIT_Test\1024\mask\*.tif")
'''
img_trial_path = r"G:\VIT_Test\1024\img"
img_dataset_name = [i for i in os.listdir(img_trial_path) if i.endswith(".TIF")]
##for i in img_dataset_name:
##    print(i)
mask_trial_path = r"G:\VIT_Test\1024\mask"
mask_dataset_name = [i for i in os.listdir(mask_trial_path) if i.endswith(".TIF")]
##for i in mask_dataset_name:
##    print(i)

X_train_name, X_val_name, X_test_name, Y_train_name, Y_val_name, Y_test_name = split_name_for_train_test(img_dataset_name, mask_dataset_name, split_size=0.3)


print("X_val")
for i in X_val_name:
    print(i)

print("Y_val")
for i in Y_val_name:
    print(i)

print("X_test")    
for i in X_test_name:
    print(i)
    
print("Y_test")
for i in Y_test_name:
    print(i)

sf_1 = r"G:\VIT_Test\1024\img"
sf_2 = r"G:\VIT_Test\1024\mask"
df_1 = r"G:\VIT_Test\1024\test_img"
df_2 = r"G:\VIT_Test\1024\test_mask"
df_3 = r"G:\VIT_Test\1024\val_img"
df_4 = r"G:\VIT_Test\1024\val_mask"

def copy(sf1, df1, X_test_name):
    for i in X_test_name:
        source_file_path = os.path.join(sf1, i)
        dest_file_path = os.path.join(df1, i)

        shutil.copy2(source_file_path, dest_file_path)

copy(sf_1, df_1, X_test_name)
copy(sf_2, df_2, Y_test_name)
copy(sf_1, df_3, X_val_name)
copy(sf_2, df_4, Y_val_name)
'''


print("loading dataset")
#profiler_client.start_profiler()
start1 = time.time()
img_dataset_arr, mask_dataset_arr, mask_colors = load_dataset(sample_image_path, sample_mask_path, num_of_samples)
end1 = time.time()
#profiler_client.stop_profiler()
print(f'{end1 - start1} seconds for loading dataset')
print(f'{(end1-start1)/60} minutes')
print("dataset loaded")
print("\n")
print("size of image dataset")
print(sys.getsizeof(img_dataset_arr))
print("size of mask dataset")
print(sys.getsizeof(mask_dataset_arr))



print("calculating number of classes")
start1 = time.time()
class_colors, num_classes = no_of_classes(mask_colors)
end1 = time.time()
print(f'{end1 - start1} seconds to calculate number of classes')
print("number of classes calculated")




print("normalizing not done")
start1 = time.time()
#master_images_dataset = normalise_images_for_training(img_dataset_arr, num_of_samples)
master_images_dataset = img_dataset_arr #not normalising data
end1 = time.time()
print(end1 - start1)
print("normalizing not done")
print(sys.getsizeof(master_images_dataset))


print("one-hot encoding")
start1 = time.time()
master_label_dataset = one_hot_encode_masks(mask_dataset_arr)
end1 = time.time()
print(f'{end1 - start1} seconds to one-hot encode masks')
print("one-hot encoding done")
print("size of master label dataset")
print(sys.getsizeof(master_label_dataset))

X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data_for_train_test(master_images_dataset, master_label_dataset, split_size=validation_split_size)


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)




#### MODEL ARCHITECTURE

def SqueezeAndExcitation(inputs, ratio=8):

    b, h, w, c = inputs.shape

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation='relu', use_bias=False)(x)
    x = Dense(c, activation='sigmoid', use_bias=False)(x)

    x = Multiply()([inputs, x])

    return x

def ASPP(inputs):
    shape = inputs.shape

    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = Conv2D(filters=256, kernel_size=1,padding='same', use_bias=False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=5, padding='same', use_bias=False)(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=10, padding='same', use_bias=False)(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=15, padding='same', use_bias=False)(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y



def DeepLabV3Plus(shape):
    """ Inputs """
    inputs = Input(shape)

    """ Pre-trained ResNet101 """
    base_model = ResNet101(weights=r"C:\Users\rcsouth\Desktop\Python and Cuda\resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_tensor=inputs)

    """ Pre-trained ResNet101 Output """
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    """ Get low-level features """
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)
    #x_b = layers.Dropout(0.2)(x_b)

    x = Concatenate()([x_a, x_b])
    x  = SqueezeAndExcitation(x, ratio=16)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layers.Dropout(0.25)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layers.Dropout(0.25)(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    """ Outputs """
    x = Conv2D(5, (1, 1), name='output_layer')(x)
    x = Activation('sigmoid')(x)

    """ Model """
    model = Model(inputs=inputs, outputs=x)
    return model


model = DeepLabV3Plus(img_shape)


model.compile(optimizer=optimiser, loss=loss_function, metrics=(['accuracy', tf.keras.metrics.MeanIoU(num_classes=num_classes)]))


# Train the model
start1 = time.time()
model_history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, Y_val))
end1 = time.time()
print(f'{end1 - start1} seconds to train the model')
print((end1-start1)/60)
print((end1-start1)/3600)


results = model.evaluate(X_test, Y_test, batch_size = batch_size)

print(results)

model.save(r"G:\VIT_Test\trial_deeplab_1024_20ep_5class_dropout_bs4_dil_5_10_15_IT2.h5")

print(model_history)
print(type(model_history))
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
plt.title("Training and Validation Accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.title("Training Loss")
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'r', label = 'Training accuracy')
plt.title("Training Accuracy")
plt.legend()


plt.show()





##i = 2644
##
##dataset_path = r"G:\rams\VIT_Test\Training_Samples_All_Reduced\512\onlytif"
##
##input_img = cv2.imread(r'G:\rams\VIT_Test\Training_Samples_All_Reduced\512\onlytif\img\img_512_154.TIF',1)
##print(input_img.shape)
##
##mask_img = cv2.imread(r'G:\rams\VIT_Test\Training_Samples_All_Reduced\512\onlytif\mask\mask_512_154.TIF',1)
##'''
##pixels_img = input_img.reshape(-1,input_img.shape[-1])
##scaled_pixels_img = np.empty_like(pixels_img, dtype=float)
##for band in range(input_img.shape[-1]):
##    scaler = MinMaxScaler()
##    scaled_band = scaler.fit_transform(pixels_img[:, band].reshape(-1,1))
##    scaled_pixels_img[:, band] = scaled_band.flatten()
##scaled_image = scaled_pixels_img.reshape(input_img.shape)
##normalized_image = scaled_image
##'''
###reshaped_img = np.expand_dims(normalized_image, axis=0)
##reshaped_img = np.expand_dims(input_img, axis=0)
##print(reshaped_img.shape)
###print(input_img.shape, normalized_image.shape, scaled_pixels_img.shape, reshaped_img.shape)
##predictions = model.predict(reshaped_img)
###predictions = model.predict(input_img)
##
##categorical_mask = np.argmax(predictions[0], axis=2)
##output_image = np.zeros((512, 512, 3), dtype = np.uint8)
##print(output_image.shape)
##for label, color in class_colors.items():
##    output_image[categorical_mask == label] = color
##
##
##
##plt.subplot(131)
##plt.imshow(input_img)
##plt.title("Input Image")
##
##plt.subplot(132)
##plt.imshow(mask_img)
##plt.title("Actual Mask")
##
##plt.subplot(133)
##plt.imshow(output_image)
##plt.title("Predicted Mask")
##
##plt.show()


