
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df_1=pd.read_csv("first_telc.csv")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''
    

    try:
        inputQuery1 = request.form['query1']
        inputQuery2 = request.form['query2']
        inputQuery3 = request.form['query3']
        inputQuery4 = request.form['query4']
        inputQuery5 = request.form['query5']
        inputQuery6 = request.form['query6']
        inputQuery7 = request.form['query7']
        inputQuery8 = request.form['query8']
        inputQuery9 = request.form['query9']
        inputQuery10 = request.form['query10']
        inputQuery11 = request.form['query11']
        inputQuery12 = request.form['query12']
        inputQuery13 = request.form['query13']
        inputQuery14 = request.form['query14']
        inputQuery15 = request.form['query15']
        inputQuery16 = request.form['query16']
        inputQuery17 = request.form['query17']
        inputQuery18 = request.form['query18']
        inputQuery19 = request.form['query19']
    except KeyError as e:
        return render_template('home.html',
                               output1=f"Error: Missing field {e}. Please fill all fields.",
                               output2="")

    try:
        monthly_charges = float(inputQuery2)
        total_charges   = float(inputQuery3)
        tenure_val      = int(inputQuery19)
        senior_citizen  = int(inputQuery1)
    except ValueError:
        return render_template('home.html',
                               output1="Error: SeniorCitizen, MonthlyCharges, TotalCharges and Tenure must be numeric.",
                               output2="",
                               query1=inputQuery1, query2=inputQuery2,
                               query3=inputQuery3, query4=inputQuery4,
                               query5=inputQuery5, query6=inputQuery6,
                               query7=inputQuery7, query8=inputQuery8,
                               query9=inputQuery9, query10=inputQuery10,
                               query11=inputQuery11, query12=inputQuery12,
                               query13=inputQuery13, query14=inputQuery14,
                               query15=inputQuery15, query16=inputQuery16,
                               query17=inputQuery17, query18=inputQuery18,
                               query19=inputQuery19)
 
    model = pickle.load(open("model.sav", "rb"))
    
    data = [[
        senior_citizen, monthly_charges, total_charges,
        inputQuery4,  inputQuery5,  inputQuery6,  inputQuery7,
        inputQuery8,  inputQuery9,  inputQuery10, inputQuery11,
        inputQuery12, inputQuery13, inputQuery14, inputQuery15,
        inputQuery16, inputQuery17, inputQuery18, tenure_val
    ]]
 
    new_df = pd.DataFrame(data, columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure'
    ])
 
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
 
    # Group tenure into bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(
        df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels
    )
 
    # Drop tenure (replaced by tenure_group)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
 
    new_df_dummies = pd.get_dummies(df_2[[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]])
 
    single      = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]
 
    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
 
    return render_template('home.html',
                           output1=o1, output2=o2,
                           query1=inputQuery1,  query2=inputQuery2,
                           query3=inputQuery3,  query4=inputQuery4,
                           query5=inputQuery5,  query6=inputQuery6,
                           query7=inputQuery7,  query8=inputQuery8,
                           query9=inputQuery9,  query10=inputQuery10,
                           query11=inputQuery11, query12=inputQuery12,
                           query13=inputQuery13, query14=inputQuery14,
                           query15=inputQuery15, query16=inputQuery16,
                           query17=inputQuery17, query18=inputQuery18,
                           query19=inputQuery19)
 
 
# FIX 4: Guard app.run() so the dev server only starts when this file
# is executed directly, not when it is imported as a module.
if __name__ == '__main__':
    app.run(debug=False)