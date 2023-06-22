#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:19:46 2019

"""



# Based on: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle





               
    

# Import pickled results of Resnet model

  # First batch
list_of_output_dicts_so_far = pickle.load( open( '/Users/jamesmanzi/Desktop/Personal/Smithsonian/list_of_output_dicts_backup_2.p', "rb" ) )

  # Second batch
list_of_output_dicts_next = pickle.load( open( '/Users/jamesmanzi/Desktop/Personal/Smithsonian/list_of_output_dicts_chunk_0_backup.p', "rb" ) )

  # Large directory of results
  
    # Get list of file names in directory
list_of_pickled_results_file_names = os.listdir('/Users/jamesmanzi/Desktop/Personal/Smithsonian/resnet_image_results')
list_of_pickled_results_file_names = [item for item in list_of_pickled_results_file_names if '.DS' not in item]

    # Uterate through names, loading pickle file for each
list_of_pickled_results = []
for item in list_of_pickled_results_file_names:
    pickle_path = '/Users/jamesmanzi/Desktop/Personal/Smithsonian/resnet_image_results/' + item
    pickled_results = pickle.load( open( pickle_path, "rb" ) )
    list_of_pickled_results.append(pickled_results)

    # Flatten list of lists
list_of_pickled_results = [item for sublist in list_of_pickled_results for item in sublist]


  # Combine first two parts with big part
all_resnet_results =   list_of_output_dicts_so_far + list_of_output_dicts_next + list_of_pickled_results
  



# PROCESS RAW RESULTS INTO DF


# Load name lookups
name_lookup_dict = pickle.load( open( '/Users/jamesmanzi/Desktop/Personal/Smithsonian/Code/name_lookup_dict.p', "rb" ) )
name_lookup_dict_2 = pickle.load( open( '/Users/jamesmanzi/Desktop/Personal/Smithsonian/Code/name_lookup_dict_2.p', "rb" ) )



    

 

# Populate prediction dataframe with relevant values from resnet model

  # Placeholder list
df_root_list = []

  # Iterate through each resent result
for resnet_result in all_resnet_results:
    
    # Extract just ouput dict
    output_dict = resnet_result[1]
    
    # Extract predictions for Bombus species
    output_df = pd.DataFrame(  list  (  zip (list(output_dict['detection_classes']), list(output_dict['detection_scores']))))
    output_df.columns = ['species', 'probability']
    output_df["species"].replace(name_lookup_dict_2, inplace=True)
      
    # Remove non-Bombus species from df
    output_df = output_df[output_df['species'].isin(list(name_lookup_dict_2.values()))]
    
    # Convert back to dict
    modified_output_dict = output_df.set_index('species').to_dict()['probability']

    # Go through output dict and convert to format with result for each of the target species
    
      # Placeholder list
    result_list = []
    
      # Iterate through each of the possible species names
    for item in list(name_lookup_dict_2.values()):
        
        # If the output dict has a value for this specieis, enter it
        if item in list(modified_output_dict.keys()):
            result_list.append(modified_output_dict[item])  
        
        # If not, eneter 0 probability
        else:
            result_list.append(0)  
        
    # Add this llist back to the overall result list
    df_root_list.append(result_list) 
        
    
  # Convert to DF
base_df = pd.DataFrame(df_root_list) 
cols = list(name_lookup_dict_2.values())  
base_df.columns = cols   
    
    



# Populate second prediction dataframe with relevant values from basic model

 # Import packages
from keras.models import load_model
from keras.preprocessing.image import load_img




  # Load basic model
basic_cnn_model = load_model('/Users/jamesmanzi/Desktop/Personal/Smithsonian//basic_cnn.h5') 

# Define function to get user selection, score and report results
def get_basic_scores(image_path, image_size = 224):
     
    # Load image
    img = load_img(image_path)
    
    # Re-size image
    img = img.resize((image_size,image_size))
      
    # Convert to numpy array
    img = np.array(img)
    
    # Divide all values by constant
    img = img / 255.0
    
    # Reshape
    img = img.reshape(1,image_size,image_size,3)
    
    # Apply neural network model
    model_pred = basic_cnn_model.predict(img)
    
    return model_pred
    
     
  # Apply basic model to each image for whihc there is also a resent result

    # Placeholder list
df_root_list_basic = []

    # Iterate through each result
for resnet_result in all_resnet_results:
    
    # Extract image path
    image_path = resnet_result[0]
    
    # Get model predictions
    model_pred = get_basic_scores(image_path)
    
    # Add this llist back to the overall result list
    df_root_list_basic.append(model_pred) 
        
    
  # Convert to DF
base_df_basic = pd.DataFrame(sum(map(list, df_root_list_basic), []))
cols_basic = ['basic_' + item for item in list(name_lookup_dict_2.values())]
base_df_basic.columns = cols_basic   



# Append basic_df to base df, to get 14 variables per image
base_df = pd.concat([base_df, base_df_basic], axis=1)



# Add image_path, species and test/train as columns
  
  # Image paths
image_paths =   [item[0] for item in all_resnet_results]
base_df['image_path'] = image_paths

  # Species
species_list = []
for item in image_paths:
    second_half = item.split('Bombus')[1].lstrip('_').split('_')[0].lower()
    species_name = 'Bombus_' + second_half
    species_name = species_name.split('/', 1)[0]
    species_list.append(species_name)
    
base_df['species'] = species_list


  # Test vs Train
test_train = []
for item in image_paths:
    if '/Test/' in item:
        test_train.append('test')
    elif '/Train/' in item:
        test_train.append('train')
    else:
        test_train.append('NA')

base_df['train_test'] = test_train




# Add box size

  # Placeholder list
box_size_list = []

  # Iterate through each resent result
for resnet_result in all_resnet_results:
    
    # Extract just ouput dict
    output_dict = resnet_result[1]
    
    # Extract predictions for Bombus species
    output_df = pd.DataFrame(  list  (  zip (list(output_dict['detection_classes']), list(output_dict['detection_scores']), list(output_dict['detection_boxes']))))
    output_df.columns = ['species', 'probability', 'detection_box']
    output_df["species"].replace(name_lookup_dict_2, inplace=True)
      
    # Remove non-Bombus species from df
    output_df = output_df[output_df['species'].isin(list(name_lookup_dict_2.values()))]
    
    # Sort by probability
    output_df = output_df.sort_values('probability', ascending = False)
    
    try:
    
        # Get coordinates of bounding box for most likely species
        coordinates = list(output_df['detection_box'])[0]
        
        # Calculate bounding box size
        box_size = (coordinates[2] - coordinates[0]) * (coordinates[3] - coordinates[1])
        
    except:
        box_size = 0
        
    
    # Add this llist back to the overall result list
    box_size_list.append(box_size) 


# Add to df
base_df['box_size'] = box_size_list
   
    
  
# Add brightness


  # Define function to get the brightness of an image
def calculate_brightness(image_path):
    image = Image.open(image_path)
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

  # Note: from https://gist.github.com/kmohrf/8d4653536aaa88965a69a06b81bcb022


  # Apply function to create variable
image_brightness_list = []

for image_path in image_paths:
    image_brightness =   calculate_brightness(image_path) 
    image_brightness_list.append(image_brightness)

# Add to df    
base_df['brightness'] = image_brightness_list
    
    
## Add colorfulness 
## note: https://sb-nj-wp-prod-1.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
#
#  # Import packages
#import cv2
#
#  # Define function to calculate sharpness
#def calculate_colorfulness(image_path):
#    
#    # Read in image
#    image = cv2.imread(image_path)
#    
#    # split the image into its respective RGB components
#    (B, G, R) = cv2.split(image.astype("float"))
# 
#	# compute rg = R - G
#    rg = np.absolute(R - G)
# 
#	# compute yb = 0.5 * (R + G) - B
#    yb = np.absolute(0.5 * (R + G) - B)
# 
#	# compute the mean and standard deviation of both `rg` and `yb`
#    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
#    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
# 
#	# combine the mean and standard deviations
#    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
#    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
# 
#	# derive the "colorfulness" metric and return it
#    return stdRoot + (0.3 * meanRoot)
#    
#    
#    
#    
#  # Apply function to create variable
#image_colorfulness_list = []
#
#for image_path in image_paths:
#    image_colorfulness =   calculate_colorfulness(image_path) 
#    image_colorfulness_list.append(image_colorfulness)
#
## Add to df    
#base_df['colorfulness'] = image_colorfulness_list




# BUILD MODEL

# Import base_df
base_df = pd.read_csv('/Users/jamesmanzi/Desktop/Personal/Smithsonian/base_df 2.csv')
del base_df['Unnamed: 0']


# Import packages
import lightgbm as lgb
global_seed = '1234'

# Drop cols
df = base_df
del df['image_path']

# Split into test and train
df_train = df[df['train_test'] == 'train']
df_test = df[df['train_test'] == 'test']

# Drop cols
del df_train['train_test']
del df_test['train_test']
  
# Break into predictors and outcome
y_train_names = list(df_train['species'])
y_test_names = list(df_test['species'])


del df_train['species']
del df_test['species']

X_train = df_train
X_test = df_test


# Convert species to numbeical representation

lgb_encoding_dict = {'Bombus terrestris' : 0,
 'Bombus vosnesenskii' : 1,
 'Bombus bimaculatus' : 2,
 'Bombus impatiens' : 3,
 'Bombus ternarius' : 4,
 'Bombus griseocollis' : 5,
 'Bombus pensylvanicus' : 6}



y_train_numbers = []

for item in y_train_names:
    number = lgb_encoding_dict[item.replace('_',' ')]
    y_train_numbers.append(number)


y_test_numbers = []

for item in y_test_names:
    number = lgb_encoding_dict[item.replace('_',' ')]
    y_test_numbers.append(number)



# Create arrays
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)


y_train = np.asarray(y_train_numbers) 
y_test = np.asarray(y_test_numbers) 



# Prepare train and test datasets
lgb_train_data = lgb.Dataset(X_train, y_train)
lgb_test_data = lgb.Dataset(X_test, y_test)


# Set parameters for LightGBMClassifier
params = {
    'objective' :'multiclass',
    'num_class' : 7,
    'learning_rate' : 0.02,
    'num_leaves' : 76,
    'feature_fraction': 0.64, 
    'bagging_fraction': 0.8, 
    'bagging_freq':1,
    'boosting_type' : 'gbdt',
    'metric': 'multi_logloss',
    'seed' : global_seed, 
    'nthread' : '-1'
}



# Fit lightgbm classifier
lgb_model = lgb.train(params, lgb_train_data)


# Test accuracy on validation data
predictions_lgb = list(lgb_model.predict(X_test))


# Spot-check results on validation sample
spot_check = pd.DataFrame(predictions_lgb)
cols = list(lgb_encoding_dict.keys())  
spot_check.columns = cols  
spot_check['Max'] = spot_check[cols].idxmax(axis=1)
 
# Add in actuals
y_test_names = [item.replace('_', ' ') for item in y_test_names]
spot_check['actual'] = y_test_names




# Add columns for matches with predictions
spot_check['MAX_correct'] = np.where(spot_check['Max'] == spot_check['actual'], 1, 0)


# Get percent correct for eac h
spot_check['MAX_correct'].mean()



# Save model
from sklearn.externals import joblib
joblib.dump(lgb_model, '/Users/jamesmanzi/Desktop/Personal/Smithsonian/integrated_model_LGB.p')


# Build feature importance plot
 
  # Create list of feature names   
feature_names = list(base_df.columns)
remove_elements = ['species','train_test']
feature_names = [item for item in feature_names if item not in remove_elements]

  # Get list of feature importances
feature_importance = lgb_model.feature_importance()

  # Convert to DF
importance_df = pd.DataFrame(list(zip(feature_names, feature_importance)))
importance_df.columns = ['Feature', 'Importance']

  # Sort DF
importance_df = importance_df.sort_values('Importance', ascending = False)

  # Create chart
#importance_df.plot.bar(x =  'Feature', y = 'Importance', fontsize = 8)


























###### SECTION OT MAKE PREDICTION FOR A NEW IMAGE    ###############



# =============================================================================
# os.chdir("/Users/jamesmanzi/Desktop/models-master/research/object_detection/")
# 
# # This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
# from object_detection.utils import ops as utils_ops
# 
# if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
#   raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
#   
# #  Environment setup  
# # %matplotlib inline
# #os.chdir('/Users/jamesmanzi/Desktop/FoundryDC/Tensorflow/models/research/')
# 
# 
#  # Object detection imports
# from utils import label_map_util
# from utils import visualization_utils as vis_util
#  
# #
# 
# # Load (frozen) Tensorflow model into memory
#   
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile("/Users/jamesmanzi/Desktop/Personal/Smithsonian/faster_rcnn_resnet101_fgvc_2018_07_19/frozen_inference_graph.pb", 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
# 
# # Loading label map
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
# 
# # Helper code
# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)
# 
# 
# # Detection examples
#   # For the sake of simplicity we will use only 2 images:
# # image1.jpg
# # image2.jpg
# # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = '/Users/jamesmanzi/Desktop/models-master/research/object_detection/test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
# 
# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)
# 
# 
#  
#  # Define function to run inference on an image
# def run_inference_for_single_image(image, graph):
#    with graph.as_default():
#      with tf.Session() as sess:
#        # Get handles to input and output tensors
#        ops = tf.get_default_graph().get_operations()
#        all_tensor_names = {output.name for op in ops for output in op.outputs}
#        tensor_dict = {}
#        for key in [
#            'num_detections', 'detection_boxes', 'detection_scores',
#            'detection_classes', 'detection_masks'
#        ]:
#          tensor_name = key + ':0'
#          if tensor_name in all_tensor_names:
#            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#                tensor_name)
#        if 'detection_masks' in tensor_dict:
#          # The following processing is only for single image
#          detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#          detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#          # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#          real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#              detection_masks, detection_boxes, image.shape[1], image.shape[2])
#          detection_masks_reframed = tf.cast(
#              tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#          # Follow the convention by adding back the batch dimension
#          tensor_dict['detection_masks'] = tf.expand_dims(
#              detection_masks_reframed, 0)
#        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
#  
#        # Run inference
#        output_dict = sess.run(tensor_dict,
#                               feed_dict={image_tensor: image})
#  
#        # all outputs are float32 numpy arrays, so convert types as appropriate
#        output_dict['num_detections'] = int(output_dict['num_detections'][0])
#        output_dict['detection_classes'] = output_dict[
#            'detection_classes'][0].astype(np.int64)
#        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#        output_dict['detection_scores'] = output_dict['detection_scores'][0]
#        if 'detection_masks' in output_dict:
#          output_dict['detection_masks'] = output_dict['detection_masks'][0]
#    return output_dict
#  
#  
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# #####   GET OUTPUT DATA FOR A LIST OF IMAGES   #######
#         
# 
#         
#     
#   # Define function to get all Bombbus species  and probabilities
# def get_output_dict(image_path):  
#     
#     
#     # Open and load image
#     image = Image.open(image_path)
#     image_np = load_image_into_numpy_array(image)
#  
#     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#     image_np_expanded = np.expand_dims(image_np, axis=0)
#   
#     # Actual detection.
#     output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
#   
#     return [image_path, output_dict]
#     
#     
# 
#     
# 
# 
# 
# # Define function to get output dicts for a list of image paths
# def get_output_dicts_list(image_path_list):
#     
#     list_of_output_dicts = []
#     
#     
#     for image_path in image_path_list:
#         
#         try:
#            output_pair =  get_output_dict(image_path)
#            list_of_output_dicts.append(output_pair)
#            
#            i = i+1
#            if i % 10 == 0:
#                print("Iterations completed: " + str(i))
#                with open('/Users/jamesmanzi/Desktop/Personal/Smithsonian/list_of_output_dicts_part_3.p', 'wb') as f:
#                    pickle.dump(list_of_output_dicts, f)
#                
#     
#         except:
#             pass
# =============================================================================

