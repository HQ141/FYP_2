import time
import os
import requests
import numpy as np
import pandas as pd
import ipfsApi
from requests.utils import requote_uri
import collections
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, Conv1D, BatchNormalization, GlobalAvgPool1D
moving_window=False
output_shape = (None, 3)
# Road surface type data classes
data_class_labels = ["dirt_road", "cobblestone_road", "asphalt_road"]
model_hash="QmaYZV8ogcjiQ3PM7VpbYtfQ5Se1xgk3VbAjYvUoEYeWUh"
def get_model_weights():
    if(os.path.exists('model_weights.hdf5')==False):
        api = ipfsApi.Client('127.0.0.1', 5001)
        file1 = open('model_weights.hdf5', 'w')
        file1.write(api.cat(model_hash))
        file1.close()
    return 
def push_to_chain(id,edgeserver,vehicle,model,blockhash):
    files = {
    "id":(None,id),
    "edgeserver":(None, edgeserver),
    "vehicle":(None, vehicle),
    "model":(None, model),
    "blockhash":(None, blockhash),
    }   

    response = requests.post('http://localhost:3333/data', files=files)
    print("test")

    return

def craft_request(malcious):
    if(malcious):
        path="../data_inputs/mal_data/"
    else:
        path="../data_inputs/correct_data/"
    return path
def modelBuilder(n_convolutional, n_features, n_outputs):
    
    model = Sequential([
        Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(n_convolutional, n_features)),
        BatchNormalization(),
        SpatialDropout1D(0.15),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        SpatialDropout1D(0.15),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalAvgPool1D(),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(n_outputs, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model, "CNN Model"

def loadWeights(model, file_path):
    model.load_weights(file_path)
def modelParameters(input_shape, output_shape):
    return {
        "n_convolutional": input_shape[1], 
        "n_features": input_shape[2], 
        "n_outputs": output_shape[1]
    }
def getDataSets(path):
    datasets = {}
    '''
    left =   pd.read_csv(os.path.join('dataset_gps_mpu_left.csv'))              #,  float_precision="high" , dtype=np.float32
    right =  pd.read_csv(os.path.join( 'dataset_gps_mpu_right.csv'))             #,  float_precision="high" , dtype=np.float32
    labels = pd.read_csv(os.path.join( 'dataset_labels.csv'), dtype=np.uint8)    #,  float_precision="high"
    '''
    left =   pd.read_csv(os.path.join(path+'10_left.csv'))              #,  float_precision="high" , dtype=np.float32
    right =  pd.read_csv(os.path.join(path+'10_right.csv'))             #,  float_precision="high" , dtype=np.float32
    datasets = {
            "left": left,
            "right": right
        }
    
    return datasets

def getFields(acc=False, gyro=False, mag=False, temp=False, speed=False, location=False, below_suspension=False, above_suspension=False, dashboard=False):
    all_fields = [
        'timestamp', 
        'acc_x_dashboard', 'acc_y_dashboard', 'acc_z_dashboard',
        'acc_x_above_suspension', 'acc_y_above_suspension', 'acc_z_above_suspension', 
        'acc_x_below_suspension', 'acc_y_below_suspension', 'acc_z_below_suspension', 
        'gyro_x_dashboard', 'gyro_y_dashboard', 'gyro_z_dashboard', 
        'gyro_x_above_suspension', 'gyro_y_above_suspension', 'gyro_z_above_suspension',
        'gyro_x_below_suspension', 'gyro_y_below_suspension', 'gyro_z_below_suspension', 
        'mag_x_dashboard', 'mag_y_dashboard', 'mag_z_dashboard', 
        'mag_x_above_suspension', 'mag_y_above_suspension', 'mag_z_above_suspension', 
        'temp_dashboard', 'temp_above_suspension', 'temp_below_suspension', 
        'timestamp_gps', 'latitude', 'longitude', 'speed'
    ]
    
    return_fields = []
    
    for field in all_fields:
            
        data_type = False
        placement = False
        
        if (speed and field == "speed"):
            placement = data_type = True
            
        if (location and (field == "latitude" or field == "longitude")):
            placement = data_type = True
        
        if (acc):
            data_type = data_type or field.startswith("acc_")
        
        if (gyro):
            data_type = data_type or field.startswith("gyro_")
            
        if (mag):
            data_type = data_type or field.startswith("mag_")
            
        if (temp):
            data_type = data_type or field.startswith("temp_")
            
        if (below_suspension):
            placement = placement or field.endswith("_below_suspension")
            
        if (above_suspension):
            placement = placement or field.endswith("_above_suspension")
            
        if (dashboard):
            placement = placement or field.endswith("_dashboard")
        
        if (data_type and placement):
            return_fields.append(field)
            
    return return_fields

def getSubSets(datasets, fields, labels=data_class_labels):
    subsets = {}
    subsets = {
            "left": datasets["left"][fields],
            "right": datasets["right"][fields]
    }
    
    return subsets

def getNormalizedData(subsets, scaler):
    normalized_sets = {}
    learn_data = pd.DataFrame() 
    for side in ["left", "right"]:
        learn_data = learn_data._append(subsets[side], ignore_index=True)
    scaler = scaler.fit(learn_data)
    del learn_data    
    normalized_sets= {
            'left':  pd.DataFrame(data=scaler.transform(subsets['left']),  columns=subsets['left'].columns),
            'right': pd.DataFrame(data=scaler.transform(subsets['right']), columns=subsets['right'].columns),
        }
                    
    return normalized_sets 

def getNormalizedDataMinMax(subsets, scaler_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=scaler_range)
    return getNormalizedData(subsets, scaler)

def getReshapedData(subsets, shape):  
    last_label=False
    shape = tuple([x for x in shape if x is not None])

    reshaped_sets = {}

    window = 300

    reshaped_sets = {};

    for side in ['left', 'right']:

            inputs = subsets[side].values
  
            inputs_reshaped = []

            for i in range(window, len(inputs) + 1):
                
                input_window = inputs[i-window:i, :]
                if moving_window or i % window == 0:
                    inputs_reshaped.append(input_window.reshape(shape))

            reshaped_sets[side] = np.array(inputs_reshaped) # inputs_reshaped
            del inputs_reshaped

    return reshaped_sets, window

def getTrainValidationSets(reshaped_sets):   
    for side in ["left", "right"]:
            inputs = reshaped_sets[side]
    return np.array(inputs)
experiment_by_placement = [
    ("Below Suspension", getFields(acc=True, gyro=True, speed=True, below_suspension=True)),
    ("Above Suspension", getFields(acc=True, gyro=True, speed=True, above_suspension=True)),
    ("Dashboard",        getFields(acc=True, gyro=True, speed=True, dashboard=True))
]

best_model = {
    "input_shape": (None, 300, 7),
    "output_shape": (None, 3),
    "file_weights": [
        {
            "exp_placement": 0,
            "exp_dataset": 0,
            "exp_placement_name": "Below Suspension",
            "exp_dataset_name": "Dataset 1",
            "file_name": r'model_weights.hdf5'
        }
    ]
}

def craft_files():
    if os.path.exists(r'./model_weights.hdf5'):
        print("File exists")
    else:
        get_model_weights()
    res = os.system("rm ../crafted_files/ev*")
    best_model_selected = best_model['file_weights'][0]
    exp_placement = best_model_selected["exp_placement"]
    input_shape = best_model["input_shape"]
    output_shape = best_model["output_shape"]
    file_weights = best_model_selected["file_name"]
    fields = experiment_by_placement[exp_placement][1]
    path=craft_request(True)
    datasets = getDataSets(path)
    subsets = getSubSets(datasets.copy(), fields, data_class_labels)
    normalized_sets = getNormalizedDataMinMax(subsets, (-1,1))

    reshaped_sets, window_size = getReshapedData(normalized_sets, input_shape)

    input = getTrainValidationSets(reshaped_sets)
    finput=input
    del subsets, normalized_sets, reshaped_sets
    model_args = modelParameters(input_shape, output_shape)
    model, model_name = modelBuilder(**model_args)
    loadWeights(model, file_weights)
    outputs=[]
    a=0
    for i in input:
        single_input = input[a].reshape(1, *input[a].shape)
        test=pd.DataFrame(np.concatenate(single_input))
        single_output = model.predict(single_input)
        predicted_class = single_output.argmax(axis=1)
        print(predicted_class[0])
        outputs.append(predicted_class[0])
        a=a+1
    a=0
    majority=collections.Counter(outputs).most_common()[0][0]
    print(majority)
    for i in finput:
        single_input = finput[a].reshape(1, *finput[a].shape)
        test=pd.DataFrame(np.concatenate(single_input))
        single_output = model.predict(single_input)
        predicted_class = single_output.argmax(axis=1)
        print(predicted_class[0])
        if(predicted_class[0]==majority):
            test['mal'] = 0
            #test['mal'] = 1
        else:
            test['mal'] = 1
        test.to_csv(f"../crafted_files/ev{a}")
        hash=files_to_ipfs(f"../crafted_files/ev{a}")
        push_to_chain(f"{a}","rsu1",f"ev{a}",model_hash,hash)
        a=a+1

def files_to_ipfs(filename):
    api = ipfsApi.Client('127.0.0.1', 5001)
    res=api.add(filename)
    print(res)
    return (res[0]['Hash'])

craft_files()

