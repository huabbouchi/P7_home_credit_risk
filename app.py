#to run :
#en mode local :dans le dossier du fichier app.py faire python app.py puis dans le navigateur aller à http://127.0.0.1:5000/

# Import all packages and libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
import pickle
from zipfile import ZipFile
from lightgbm import LGBMClassifier


app= Flask(__name__)
@app.route('/')
def home():
    z = ZipFile("X_dashboard.zip")
    dataframe = pd.read_csv(z.open('X_dashboard.csv'), encoding ='utf-8')
    all_id_client = list(dataframe['SK_ID_CURR'].unique())

    return render_template('index.html', customers=all_id_client)




@app.route('/predict', methods=['POST', 'GET'])
def predict():
    z = ZipFile("X_dashboard.zip")
    dataframe = pd.read_csv(z.open('X_dashboard.csv'), encoding ='utf-8')
    all_id_client = list(dataframe['SK_ID_CURR'].unique())
    prediction = 'please select a customer ID'

    model = pickle.load(open('LGBMClassifier_auc_score.pkl', 'rb'))
    seuil = 0.45
    ID = request.form.get('customer_select')
    if ID == '':
        pass
    else:

        ID = int(ID)

        X = dataframe[dataframe['SK_ID_CURR'] == ID]
        X = X.drop(['SK_ID_CURR'], axis=1)

        #data = df[df.index == comment]
        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Credit Not Accorded for customer: "
        else:
            prediction = "Credit Accorded for customer: "


    return render_template('index.html', customers=all_id_client, prediction_text=prediction, client_id=ID)

# Define endpoint for flask
app.add_url_rule('/predict', 'predict', predict)


# Run app.
# Note : comment this line if you want to deploy on heroku
#app.run()
# app.run(debug=True)
