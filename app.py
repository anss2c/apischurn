from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import json
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    modelpredic = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    dataTesting = pd.read_csv("data/test1.csv")
    dataTesting.area_code = dataTesting.area_code.map({'area_code_415':415,'area_code_408':408,'area_code_510':510})
    dataTesting = dataTesting.replace({'voice_mail_plan':{'yes':1,'no':0}})
    dataTesting = dataTesting.replace({'international_plan':{'yes':1,'no':0}})
    dataTesting = dataTesting.replace({'churn':{'yes':1,"no":0}})
    dataTesting.state = dataTesting.state.astype('category')
    col_drop = ['total_day_minutes','total_night_minutes','total_eve_minutes','total_intl_minutes']
    dataTesting = dataTesting.drop(columns = col_drop,axis = 1)
    X_new_test = dataTesting.drop(['id','state'],axis = 1)
    scaler = StandardScaler()
    X_test_std = scaler.fit_transform(X_new_test)
    df_test = pd.DataFrame(X_test_std,columns = dataTesting.columns[2:])
    predict_test = modelpredic.predict(df_test)
    pred_value_1 = predict_test.round()
    df = pd.DataFrame({'id':dataTesting.id,'state':dataTesting.state,"churn":pred_value_1})
    result = df.to_json(orient="split")
    parsed = json.loads(result)
    json.dumps(parsed, indent=4)
    return result

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':

    #modelpredic = joblib.load("model.pkl") # Load "model.pkl"
    #print ('Model loaded')
    #model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    #print ('Model columns loaded')

    app.run(debug=True)
    # app.run(threaded=True, port=5000)
