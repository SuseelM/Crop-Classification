### import all libraries necessary
import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
###import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import layers
###from tensorflow.keras import mixed_precision as mixed_precision
from tensorflow.keras.regularizers import l2
###from tensorflow.python.profiler import profiler_client
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, add, ZeroPadding2D
from keras import layers, Model

"""
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
"""

###tf.keras.mixed_precision.set_global_policy("mixed_float16")

###tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
'''
with tf.device('/cpu:0'):
    x = tf.convert_to_tensor(x, np.float32)
    y = tf.convert_to_tensor(y, np.float32)
'''
'''
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)
'''

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
fh = open(r"G:\VIT_Test\trial\sample_1024.txt", 'r')
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





##
##img_path = 'G:/VIT_Test/512_split/img/train_img'
##mask_path = 'G:/VIT_Test/512_split/mask/train_mask'
##val_img_path = 'G:/VIT_Test/512_split/img/val_img'
##val_mask_path = 'G:/VIT_Test/512_split/mask/val_mask'
img_path = os.path.dirname(sample_image_path)
mask_path = os.path.dirname(sample_mask_path)
print(img_path)

#OPTIONAL PARAMETERS

try:
    num_of_samples = int(parameters['number_of_samples'])  #number of samples
except:
    images = len(os.listdir(img_path))
    num_of_samples = images
#model_choice = 'Resnet50' #user choice
include_weights = 'yes' #user choice
try:
    validation_split_size = float(parameters['validation_size'])
except:
    validation_split_size = 0.3
try:
    dropout_percentage = float(parameters['dropout_percentage'])
except:
    dropout_percentage = 0.25 #user choice

try:
    learning_rate = float(parameters['learning_rate'])
except:
    learning_rate = 0.001 #user choice
stride_len = (2,2)
adam_optimiser = tf.keras.optimizers.Adam(learning_rate = learning_rate)
SGD_optimiser = tf.keras.optimizers.SGD(learning_rate = learning_rate)
optimiser_choice = 'adam' #user choice
if optimiser_choice=='adam':
    optimiser = adam_optimiser
elif optimiser_choice=='SGD':
    optimiser = SGD_optimiser
##loss_function = 'categorical_crossentropy'


metrics = ['accuracy', 'jaccard_coef'] #user choice
try:
    batch_size = int(parameters['batch_size'])
except:
    batch_size = 3 #user choice
try:
    num_epochs = int(parameters['num_epochs'])
except:
    num_epochs = 10#user choice
steps_per_epoch = num_of_samples//batch_size

img_height = 1024

img_shape = (img_height,img_height,3)




# making code modular in structure by using functions

def load_dataset(sample_image_path, sample_mask_path, num_of_samples):
    img_dataset = []
    mask_dataset = []
    digit_list = ['0','1','2','3','4','5','6','7','8','9','(',')']
    
    img_path = os.path.dirname(sample_image_path)
    img_name, img_extension = os.path.split(sample_image_path)[-1].split('.')
    
    for i in reversed(range(len(img_name))):
        if img_name[i] not in digit_list:
            img_basename = img_name[:i]
            break
    #print(img_name, len(img_name), img_basename)
    #print(img_name, img_basename)
    mask_path = os.path.dirname(sample_mask_path)
    #print(mask_path)
    mask_name, mask_extension = os.path.split(sample_mask_path)[-1].split('.')
    for i in reversed(range(len(mask_name))):
        if mask_name[i] not in digit_list:
            mask_basename = mask_name[:i]
            break


    mask_colors = []
    ctr = 0
    
    for i in os.listdir(img_path):
        if i.endswith("TIF")==False:
            continue
        if ctr<num_of_samples:
            img_file_name = i.split('.')[0]
            #print(i, img_file_name)
            img_file = img_file_name+'.'+img_extension
            mask_file_name = img_file_name.replace(img_basename, mask_basename)
            #print(mask_basename, mask_file_name)
            mask_file = mask_file_name+'.'+mask_extension
            #print(img_file, mask_file)
            
            img = cv2.imread(os.path.join(img_path,img_file))
            #print(os.path.join(mask_path,mask_file))
            mask = cv2.imread(os.path.join(mask_path,mask_file))
            #print(mask)
            # failsafe to ignore files which do not exist or do not read or not in proper format
            if img is None or mask is None:
                print("one of both files is not readable, skipping both")
                print(f"{img_file} and {mask_file}")
                continue
            # to find the unique colors in each mask image
            uniq_per_mask = np.unique(mask, axis=0)
            flattened_image_array = uniq_per_mask.reshape(-1, uniq_per_mask.shape[-1])
            unique_colors = set(tuple(color) for color in flattened_image_array)
            unique_colors = [list(color_tuple) for color_tuple in unique_colors]
            mask_colors.append(unique_colors)
            # create datasets
            img_dataset.append(img)
            mask_dataset.append(mask)
            #print("\n\n")
            ctr = ctr+1
    
    img_dataset_arr = np.array(img_dataset)
    del img_dataset
    mask_dataset_arr = np.array(mask_dataset)
    del mask_dataset
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


def dice_coef(y_true, y_pred, smooth = 100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def generalized_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
    return final_coef_value



##def inferencing(model, file_number):
##    input_img = cv2.imread(f'{dataset_path}/images/image_{file_number}.tif',1)
##    mask_img = cv2.imread(f'{dataset_path}/masks/mask_{file_number}.tif',1)
##    # input_img has shape (height, width, bands)
##
##    # Reshape the 3D image into a 2D array where each row is a pixel and each column is a band
##    pixels_img = input_img.reshape(-1, img.shape[-1])
##
##    # Initialize an empty array to store the scaled pixel values
##    scaled_pixels_img = np.empty_like(pixels_img, dtype=float)
##
##    # Apply Min-Max scaling to each band separately
##    for band in range(img.shape[-1]):
##        scaler = MinMaxScaler()
##        scaled_band = scaler.fit_transform(pixels_img[:, band].reshape(-1, 1))
##        scaled_pixels_img[:, band] = scaled_band.flatten()
##
##    # Reshape the scaled pixels_img back to the original image shape
##    scaled_image = scaled_pixels_img.reshape(img.shape)
##
##    normalised_image = scaled_image
##
##    reshaped_img = np.expand_dims(normalised_image, axis=0)
##    predictions = model.predict(reshaped_img)
##    return predictions
##


print("loading dataset")
#profiler_client.start_profiler()
start1 = time.time()
img_dataset_arr, mask_dataset_arr, mask_colors = load_dataset(sample_image_path, sample_mask_path, num_of_samples)
end1 = time.time()
#profiler_client.stop_profiler()
print(end1 - start1)
print((end1-start1)/60)
print("dataset loaded")
print(sys.getsizeof(img_dataset_arr))
print(sys.getsizeof(mask_dataset_arr))



print("calculating number of classes")
start1 = time.time()
class_colors, num_classes = no_of_classes(mask_colors)
end1 = time.time()
print(end1 - start1)
print("number of classes calculated")




print("normalizing not done")
start1 = time.time()
#master_images_dataset = normalise_images_for_training(img_dataset_arr, num_of_samples)
master_images_dataset = img_dataset_arr
end1 = time.time()
print(end1 - start1)
print("normalizing not done")
print(sys.getsizeof(master_images_dataset))


print("one-hot encoding")
start1 = time.time()
master_label_dataset = one_hot_encode_masks(mask_dataset_arr)
end1 = time.time()
print(end1 - start1)
print("one-hot encoding done")
print(sys.getsizeof(master_label_dataset))

X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data_for_train_test(master_images_dataset, master_label_dataset, split_size=validation_split_size)


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)


##loss_function = generalized_dice_loss

loss_function = 'categorical_crossentropy'



 #MODEL DEFINITION

if include_weights=='yes':
    base_model = ResNet101(weights=r"C:\Users\rcsouth\Desktop\Python and Cuda\resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_shape=img_shape)
elif include_weights=='no':
    base_model = ResNet101(weights=None , include_top=False, input_shape=img_shape)


    
s1 = base_model.get_layer("input_1").output  #(img_width x img_height)
s2 = base_model.get_layer("conv1_relu").output #(None, 128, 128, 64)
s3 = base_model.get_layer("conv2_block3_out").output #((None, 64, 64, 256)
s4 = base_model.get_layer("conv3_block4_out").output #(None, 32, 32, 512)
print(s1.shape, s2.shape, s3.shape, s4.shape)



##, kernel_regularizer=l2(0.01)


b1 = base_model.get_layer("conv4_block23_out").output
print(b1.shape)

u6 = layers.Conv2DTranspose(img_height, (2,2), strides=stride_len, padding="same")(b1)
u6 = layers.concatenate([u6, s4])
c6 = layers.Conv2D(img_height, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(u6)
c6 = layers.Dropout(dropout_percentage)(c6)
c6 = layers.BatchNormalization()(c6)
c6 = layers.Conv2D(img_height, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(c6)


u7 = layers.Conv2DTranspose(img_height/2, (2,2), strides=stride_len, padding="same")(c6)
u7 = layers.concatenate([u7, s3])
c7 = layers.Conv2D(img_height/2, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(u7)
c7 = layers.Dropout(dropout_percentage)(c7)
c7 = layers.BatchNormalization()(c7)
c7 = layers.Conv2D(img_height/2, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(c7)


u8 = layers.Conv2DTranspose(img_height/4, (2,2), strides=stride_len, padding="same")(c7)
u8 = layers.concatenate([u8, s2])
c8 = layers.Conv2D(img_height/4, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(u8)
c8 = layers.Dropout(dropout_percentage)(c8)
c8 = layers.BatchNormalization()(c8)
c8 = layers.Conv2D(img_height/4, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(c8)


u9 = layers.Conv2DTranspose(img_height/8, (2,2), strides=stride_len, padding="same")(c8)
u9 = layers.concatenate([u9, s1])
c9 = layers.Conv2D(img_height/8, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(u9)
c9 = layers.Dropout(dropout_percentage)(c9)
c9 = layers.BatchNormalization()(c9)
c9 = layers.Conv2D(img_height/8, (3,3), activation="relu", kernel_initializer="he_normal", padding="same", kernel_regularizer=l2(0.01))(c9)

print(u6.shape, u7.shape, u8.shape, u9.shape)

outputs = layers.Conv2D(num_classes, (1,1), activation="softmax")(c9)

model = Model(inputs=base_model.input, outputs=outputs)

##model_path= r"G:\VIT_Test\trial_updated_1024_10ep_5class_dropout_reg.h5"
##model = tf.keras.models.load_model(model_path)


model.compile(optimizer=optimiser, loss=loss_function, metrics= ['accuracy', tf.keras.metrics.IoU(num_classes=num_classes, target_class_ids = class_colors) ])

start1 = time.time()
model_history = model.fit(X_train,Y_train, validation_data=(X_val,Y_val) , batch_size=batch_size, epochs=num_epochs)
end1 = time.time()
print(end1 - start1)


results = model.evaluate(X_test, Y_test, batch_size = batch_size)

print(results)

model.save(r"G:\VIT_Test\trial_updated_1024_RS101_10ep_5class_reg_only.h5")


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

plt.figure()

plt.plot(epochs, loss, 'b', label = 'Validation loss')
plt.title("Validation Loss")
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'b', label = 'Validation accuracy')
plt.title("Validation Accuracy")
plt.legend()

plt.show()








#############################################################################
##model_path=r"G:\VIT_Test\trial_updated_1024.h5"
##model = tf.keras.models.load_model(model_path, custom_objects= {"generalized_dice_loss": generalized_dice_loss})

##i = 100
##
##dataset_path = r"G:\VIT_Test\1024"
##
##input_img = cv2.imread(f'{dataset_path}/img/img_1024_ ({i}).tif',1)
##
##
##mask_img = cv2.imread(f'{dataset_path}/mask/mask_1024_ ({i}).tif',1)
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
###print(input_img.shape, normalized_image.shape, scaled_pixels_img.shape, reshaped_img.shape)
##predictions = model.predict(reshaped_img)
###predictions = model.predict(input_img)
##
##categorical_mask = np.argmax(predictions[0], axis=2)
##output_image = np.zeros((1024, 1024, 3), dtype = np.uint8)
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
##
