from flask import Flask,render_template,request
import joblib
import os
import numpy as np
import pickle

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("hometemp.html")

@app.route("/result",methods=['POST','GET'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease=int(request.form['heart_disease'])
    ever_married=int(request.form['ever_married'])
    work_type=int(request.form['work_type'])
    Residence_type=int(request.form['Residence_type'])
    avg_glucose_level=float(request.form['avg_glucose_level'])
    bmi=float(request.form['bmi'])
    smoking_status=int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scalar_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\scalar.pkl')
    scalar=None
    with open(scalar_path,'rb') as scalar_file:
        scalar=pickle.load(scalar_file)

    x=scalar.transform(x)

    model = int(request.form['model_sel'])
    if model == 1:
        model_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\svm.sav')
        final=joblib.load(model_path)

        y_pred=final.predict(x)

        if y_pred==0:
            return render_template('nostroketemp.html')
        else:
            return render_template('stroketemp.html')
    if model == 2:
        model_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\knn.sav')
        final=joblib.load(model_path)

        y_pred=final.predict(x)

        if y_pred==0:
            return render_template('nostroketemp.html')
        else:
            return render_template('stroketemp.html')
    if model == 3:
        model_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\lr.sav')
        final=joblib.load(model_path)

        y_pred=final.predict(x)

        if y_pred==0:
            return render_template('nostroketemp.html')
        else:
            return render_template('stroketemp.html')
    if model == 4:
        model_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\dt.sav')
        final=joblib.load(model_path)

        y_pred=final.predict(x)

        if y_pred==0:
            return render_template('nostroketemp.html')
        else:
            return render_template('stroketemp.html')
    if model == 5:
        model_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\ab.sav')
        final=joblib.load(model_path)

        y_pred=final.predict(x)

        if y_pred==0:
            return render_template('nostroketemp.html')
        else:
            return render_template('stroketemp.html')
    if model == 6:
        model_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\bnb.sav')
        final=joblib.load(model_path)

        y_pred=final.predict(x)

        if y_pred==0:
            return render_template('nostroketemp.html')
        else:
            return render_template('stroketemp.html')
    if model == 7:
        model_path=os.path.join('C:\\Users\\SANDEEP REDDY CHALLA\\OneDrive\\Desktop\\Early-Detection-of-Brain-Stroke-main\\Early-Detection-of-Brain-Stroke-main','models\\rf.sav')
        final=joblib.load(model_path)

        y_pred=final.predict(x)

        if y_pred==0:
            return render_template('nostroketemp.html')
        else:
            return render_template('stroketemp.html')

if(__name__)=="__main__":
    app.run(debug=True)
