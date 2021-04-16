from sklearn import datasets
from joblib import load
import numpy as np
import json

#Load my model

my_model = load('heart_model.pkl')

class_names = ['Alive','Dead']

def my_prediction(id):
    dummy = np.array(id)
    dummyT = dummy.reshape(1,-1)
    r = dummy.shape
    t = dummyT.shape
    r_str = json.dumps(r)
    t_str = json.dumps(t)
    prediction = my_model.predict(dummyT)
    #name = class_names[prediction]
    #name = name.tolist()
    #name_str = json.dumps(prediction)
    name = class_names[ int(prediction)]
    str = [t_str, r_str, name]
    return str
