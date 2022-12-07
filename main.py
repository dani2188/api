#API endpoints

# Library imports
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

import lime 
from lime import lime_tabular
import streamlit.components.v1 as components


# Create app and model objects
app = FastAPI()

df= pd.read_csv('X_test_sample.csv')
df.set_index('sk_id_curr',inplace=True)
model = joblib.load('lgbm.joblib')


user_lst= list(map(int, df.index.to_list()))

class Client(BaseModel):
    user_id: int



#  Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the prediction with the confidence

# 189979,184435,395190,193313,



@app.post('/predict')


def get_prediction(client: Client):
        
        client= client.dict()
        user_id= client['user_id']
        data_in = df.loc[[user_id]]
        prediction = model.predict(data_in)
        probability = model.predict_proba(data_in).max()
        
        return {
        'prediction': prediction[0],
        'probability': probability
         }
     


# run API with uvicorn
if __name__ == '__main__':
        uvicorn.run(app,host='127.0.0.1',port= 8000)
        
        
# Run the fastAPI on local PC:     uvicorn main:app --reload

# Access and see the app:       http://127.0.0.1:8000/docs

