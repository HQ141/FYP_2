import os
import collections
import glob
import requests
import ipfsApi
import numpy as np
import pandas as pd
from requests.utils import requote_uri
import collections
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, Conv1D, BatchNormalization, GlobalAvgPool1D


model_hash="QmaYZV8ogcjiQ3PM7VpbYtfQ5Se1xgk3VbAjYvUoEYeWUh"
def push_to_chain(id,edgeserver,vehicle,model,blockhash):
    files = {
    "id":(None,id),
    "edgeserver":(None, edgeserver),
    "vehicle":(None, vehicle),
    "model":(None, model),
    "blockhash":(None, blockhash),
    }   
    response = requests.post('http://localhost:3333/data', files=files)
    return

def get_files_ipfs():
    
    return
def files_to_ipfs(filename):
    api = ipfsApi.Client('127.0.0.1', 5001)
    res=api.add(filename)
    print(res)
    return (res[0]['Hash'])
def recraft_inputs():
    if os.path.exists(r'model_weights.hdf5'):
        print("File exists")
    else:
        get_model_weights()
    best_model_selected = best_model['file_weights'][0]
    input_shape = best_model["input_shape"]
    output_shape = best_model["output_shape"]
    file_weights = best_model_selected["file_name"]
    model_args = modelParameters(input_shape, output_shape)
    model, model_name = modelBuilder(**model_args)
    loadWeights(model, file_weights)

    x = [os.path.join(r,file) for r,d,f in os.walk("../crafted_files") for file in f]
    x.sort()
    outputs=[]
    a=0
    for file in x:
        raw_input=pd.read_csv(file)
        input=raw_input.drop('mal', axis=1)
        input = input.iloc[: , 1:]
        input=input.to_numpy()
        input=np.array(input)
        input = input.reshape(-1, 300, 7)
        single_output = model.predict(input)
        predicted_class = single_output.argmax(axis=1)
        outputs.append(predicted_class[0])
        a=a+1
    a=0
    majority=collections.Counter(outputs).most_common()[0][0]
    for file in x:
        raw_input=pd.read_csv(file)
        input=raw_input.drop('mal', axis=1)
        input = input.iloc[: , 1:]
        input=input.to_numpy()
        input=np.array(input)
        input = input.reshape(-1, 300, 7)
        single_output = model.predict(input)
        predicted_class = single_output.argmax(axis=1)
        print(predicted_class[0])
        if(predicted_class[0]==majority):
            raw_input['mal'] = 0
        else:
            raw_input['mal'] = 1
        raw_input.to_csv(f"{file}")  
        a=a+1 
def update_rep_ev(ev):
    files = {
    "id":(None,ev),
    }   

    response = requests.post('http://localhost:3333/updaterep', files=files)
    return
def update_rep_rsu(rsu):
    files = {
    "id":(None,rsu),
    }   

    response = requests.post('http://localhost:3333/updaterep', files=files)
    recraft_inputs()
    return
def get_model_weights():
    return
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

def verify_inputs():
    if os.path.exists(r'./model_weights.hdf5'):
        print("File exists")
    else:
        get_model_weights()
    best_model_selected = best_model['file_weights'][0]
    input_shape = best_model["input_shape"]
    output_shape = best_model["output_shape"]
    file_weights = best_model_selected["file_name"]
    model_args = modelParameters(input_shape, output_shape)
    model, model_name = modelBuilder(**model_args)
    loadWeights(model, file_weights)
    get_files_ipfs()
    x = [os.path.join(r,file) for r,d,f in os.walk("../crafted_files") for file in f]
    x.sort()
    prev_class=[]
    outputs=[]
    a=0
    for file in x:
        raw_input=pd.read_csv(file)
        prev_class.append(raw_input['mal'].max())
        input=raw_input.drop('mal', axis=1)
        input = input.iloc[: , 1:]
        input=input.to_numpy()
        input=np.array(input)
        input = input.reshape(-1, 300, 7)
        single_output = model.predict(input)
        predicted_class = single_output.argmax(axis=1)
        outputs.append(predicted_class[0])
        a=a+1
    a=0
    wr_class=[]
    majority=collections.Counter(outputs).most_common()[0][0]
    for file in x:
        raw_input=pd.read_csv(file)
        input=raw_input.drop('mal', axis=1)
        input = input.iloc[: , 1:]
        input=input.to_numpy()
        input=np.array(input)
        input = input.reshape(-1, 300, 7)
        single_output = model.predict(input)
        predicted_class = single_output.argmax(axis=1)
        if((predicted_class[0]==majority) and prev_class[a]!=0):
            wr_class.append(file)
        if((predicted_class[0]!=majority) ):
            if(prev_class[a]==0):
                wr_class.append(file)
            _file = file.split("/")
            update_rep_ev(_file[-1])    
        a=a+1
    if(wr_class):
        update_rep_rsu("rsu1")
    return wr_class
mal_input=verify_inputs()
for i in mal_input:
    hash=files_to_ipfs(i)
    _file = i.split("/")
    push_to_chain(f"{i[-1]}","rsu1",f"{i[-1]}",model_hash,hash)