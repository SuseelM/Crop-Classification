txt file directory
sample_512 (For UNet model of patch size 512)
sample_1024 (For UNet model of patch size 1024)
deeplab_inputs_1024 (For DeepLab model of patch size 1024)
deeplab_inputs_512 (For DeepLab model of patch size 512)

Image/Mask name format : <basename>_(i).tif

Step 1: Edit the required model code and its corresponding txt file with the input file path, text file path, patch size, batch size and other parameters and run using Python IDLE 3.10
Eg: UNet_RS101_512 will have the UNet architecture with ResNet 101 Backbone and will accept the patch size of 512. Its corresponding txt file would be sample_512 

Step 2: Make necessary changes in the python code to update the model save location and its file name and type

Step 3: Matplotlib outputs are not saved automatically and have to be saved manually after each training process

Step 4: Save the IDLE results for future reference and to obtain class colors for further inferencing

Step 5: Copy the class colors obtained in the previous step to the class_colors.txt file to use in inferencing_code.py

Class Colors need to be hard coded in the codes used for Step 6 and 7

Step 6: Run the inferencing_code.py by changing the necessary parameters and call the repeat_inf(i, model) function in IDLE to generate predictions for given input patch 'i'.
	If the input folder contains 'n' images, the function call can be looped using:-
	
	for k in range(1, n+1):
		repeat_inf(k, model)

	This will generate the predicted patches for all the input values available in the input folder.

Step 7: Run the conf_mat.py by changing the necessary parameters and call the multiple(i, n) function in IDLE to generate the confusion matrix, accuracy and IoU values for the Input Data
 
	If the input folder contains 'n' images, the function call can be looped using:-
	
	multiple(1, n)


Step 8: Run the large.py file to predicted the a large patch of image by changing the necessary parameters.
	
	Change the maximum limit of pixels in CV, if total number of pixels exceed 2^30.
	Give the right amount of Stride and Hard Code the Height and Width of the input image.
	
