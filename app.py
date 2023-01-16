import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
## Loading the regression model
regmodel=pickle.load(open('regmodel.pkl','rb')) 
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predic_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    data_val=np.array(list(data.values())).reshape(1,-1)
    print(data_val)
    new_data=scalar.transform(data_val)
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
