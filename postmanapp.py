import json
import pickle
from flask import Flask, render_template,request, app,jsonify, url_for
import numpy as np
import pandas as pd

app=Flask(__name__)

# Load the model
bst_model=pickle.load(open("model.pkl",'rb'))

@app.route("/predict_api",methods=['POST'])

def predict_api():
    #input_features=[float(x) for x in request.form.values()]
    data_json=request.json['data']

    print(type(data_json))
    data_json_list=np.array(list(data_json.values())).astype(float).reshape(1,-1)
    output=bst_model.predict(data_json_list)
    def zen(x):
        if x==1.0:
            s='Yes'
        else:
            s='No'
        return s

    Answer=zen(output[0])
    
    return jsonify(Answer)

if __name__=="__main__":
    app.run(debug=True)
