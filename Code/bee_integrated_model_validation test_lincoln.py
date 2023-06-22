#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:19:46 2019

@author: jamesmanzi
"""



# CREATE BASE_DF USING RESULTS OF RESENT OBJECT DETECTION MODEL


# Based on: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

import pandas as pd
import numpy as np
from PIL import Image
import pickle



# Load all resnet results
all_resnet_results = pickle.load( open( '/Users/jamesmanzi/Desktop/Personal/Smithsonian/all_resnet_results_lincoln.p', "rb" ) )  # CHANGE DIRECTORY

# Load name lookups
name_lookup_dict_2 = pickle.load( open( '/Users/jamesmanzi/Desktop/Personal/Smithsonian/Code/name_lookup_dict_2.p', "rb" ) )   # CHANGE DIRECTORY


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
    
    



# CREATE BASE_DF_BASIC USING RESULTS OF WHOLE IMAGE CNN MODEL

 # Import packages
from keras.models import load_model
from keras.preprocessing.image import load_img


  # Load basic model
basic_cnn_model = load_model('/Users/jamesmanzi/Desktop/Personal/Smithsonian/basic_cnn.h5') # CHANGE DIRECTORY

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
    
     
  # Apply basic model to each image for which there is also a resent result

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





# COMNINE THE TWO DATAFRAMES AND ADD IDENTIFIER COLUMNS

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


  # Test vs Train  NOTE terminology 'train' from here forward = modeling and 'test' = the validation sample
test_train = []
for item in image_paths:
    if 'validation' in item:
        test_train.append('test')
    else:
        test_train.append('train')

base_df['train_test'] = test_train






# ADD ADDIONAL VARIABLES FOR BOX SIZE AND BRIGHTNESS

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
    
    

base_df.to_csv('/Users/jamesmanzi/Desktop/Personal/Smithsonian/base_df_lincoln.csv')




# BUILD MODEL

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


# Get percent correct for each
spot_check['MAX_correct'].mean()



# Save model
from sklearn.externals import joblib
joblib.dump(lgb_model, '/Users/jamesmanzi/Desktop/Personal/Smithsonian/integrated_model_LGB_final.p')   # CHANGE DIRECTORY


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
importance_df.plot.bar(x =  'Feature', y = 'Importance', fontsize = 8)
  
  
  
  
 # Create alternative baseline for comparison: Just use resent object detection model

  # Define function to get category and probability
def get_species_resent(resnet_result, names = name_lookup_dict_2):
    
    # Extract just ouput dict
    output_dict = resnet_result[1]
  
    # Extract predictions for Bombus species
    output_df = pd.DataFrame(  list  (  zip (list(output_dict['detection_classes']), list(output_dict['detection_scores']))))
    output_df.columns = ['species', 'probability']
    output_df["species"].replace(names, inplace=True)
  
    # Remove non-Bombus species from df
    output_df = output_df[output_df['species'].isin(list(names.values()))]
  
    # Get most probable result
    output_df = output_df.sort_values('probability', ascending=False)
    best_estimate = output_df.iloc[[0]].values.tolist()[0]

    return best_estimate[0]



  # Get resent predictions
resent_preds = []

for resnet_result in all_resnet_results_validation:
    
    try:
        best_estimate = get_species_resent(resnet_result)
        resent_preds.append(best_estimate)
    except:
        resent_preds.append('NA')
 
  
  # Add to df
spot_check['resent_pred'] = resent_preds
    
  # Add columns for matches with predictions
spot_check['resnet_correct'] = np.where(spot_check['resent_pred'] == spot_check['actual'], 1, 0)


  # Get percent correct for eac h
spot_check['resnet_correct'].mean()    
    
    
    
    
    
    
    
    
    
