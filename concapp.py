import pandas as pd, numpy as np
import pickle
import flask
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
@app.route('/')
def Hm():
    return render_template('concindex.html')

model = pickle.load(open('pipeline_conc.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    x = [int(x) for x in request.form.values()]
    x_df = pd.DataFrame(x)
    x_df = x_df.T
    x_df.columns=['Cement component 1', 'Blast Furnace Slag component 2',
       'Fly Ash component 3', 'Water  component 4',
       'Superplasticizer component 5', 'Coarse Aggregate component 6',
       'Fine Aggregate component 7', 'Age day']

    prediction = model.predict(x_df)
    
    return render_template('concpredict.html',prediction_text= 'Compressive strength prediction is ' + format(prediction) + 'MPa')

if __name__ == "__main__":
    app.run(debug=True)
