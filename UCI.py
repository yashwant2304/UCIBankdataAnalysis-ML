# -*- coding: utf-8 -*-
"""
Created on Wed May 26 23:17:37 2021
This application is made up for the prediction of bank term deposit subscription of the customers.
@author: Yashwant Bhaidkar
"""
from flask import Flask, render_template, request
#import jsonify
#import requests
import pickle
import numpy as np
#import sklearn
from scipy.sparse import hstack
app = Flask(__name__)

@app.route('/',methods=['GET'])
def Home():
    return render_template('UCI.html')


@app.route("/predict", methods=['POST','GET'])
    
def predict():
    if request.method == 'POST':
        #if (len(i)==9):
         #   print('valid input size')
        #else:
         #   print('parameter size of input is incorrect.')
          #  break
            #vectorizer for job
        vectorizer = pickle.load(open("vectorizer_job.pickle", "rb"))
        input = [request.form['Job_type']]
        val_job = vectorizer.transform(input)
        #vctorizer for contact
        vectorizer2 = pickle.load(open("vect_contact_contact.pickle", "rb"))
        input = [request.form['contact_type']]
        val_contact = vectorizer2.transform(input)
        #housing loan
        if request.form['home_loan'] == 'yes':
            val_housing = 1
        else:
            val_housing = 0 
        val_housing = np.array(val_housing)
        #val_housing = val_housing.reshape(-1,1)
        #personl_loan
        if request.form['personal_loan'] == 'yes':
            val_personal = 1
        else:
            val_personal = 0 
        val_personal = np.array(val_personal)
        val_personal = val_personal.reshape(1,-1)
        #day_of_month
        input = int(request.form['date'])
        if (input>0 and input<=31):
            day = input
        else:
            print('date day is not valid')
            #break
        day = np.array(day)
        day = day.reshape(1,-1)
        #number_of_calls_given
        #print('reached')
        encoder_campaign = pickle.load(open("encoder_campaign.pickle", "rb"))
        input = request.form['campaign_calls']
        input = np.array(input)
        input = input.reshape(1,-1)
        number_of_calls_given = encoder_campaign.transform(input)
        #number_of_calls_given = np.array(number_of_calls_given)
        #number_of_calls_given = number_of_calls_given.reshape(1,-1)
        #balance
        standard_balance = pickle.load(open("standardscaler_balance.pickle", "rb"))
        input = request.form['balance']
        input = np.array(input)
        input = input.reshape(1,-1)
        balance = standard_balance.transform(input)
        #balance = np.array(balance)
        #balance = balance.reshape(1,-1)
        #call_duration_sec
        vect_duration = pickle.load(open("standardscaler_duration.pickle", "rb"))
        input = request.form['Duration']
        input = np.array(input)
        input = input.reshape(1,-1)
        call_duration_sec = vect_duration.transform(input)
        # call_duration_sec = np.array(call_duration_sec)
        # call_duration_sec = call_duration_sec.reshape(1,-1)
        #day_difference_last_call
        vect_pdays= pickle.load(open("standardscaler_previousdays.pickle", "rb"))
        input = request.form['previous']
        input = np.array(input)
        input = input.reshape(1,-1)
        day_difference_last_call = vect_pdays.transform(input)
        # day_difference_last_call = np.array(day_difference_last_call)
        # day_difference_last_call = day_difference_last_call.reshape(1,-1)
        processed_input = hstack((val_job,val_contact,val_housing,val_personal,day,number_of_calls_given,balance,call_duration_sec,day_difference_last_call)).tocsr()
        model = pickle.load(open("best_svm_model_UCI.pickle", "rb"))
        #s = processed_input.todense()
        #print(s)
        #predict = 2
        predict = model.predict(processed_input)
        #print(type(predict))
        a = predict.tolist()
        predict = int(a[0])
        print(predict)
        if predict == 0:
            print('success0')
            return render_template('UCI.html',prediction_text="Customer will not subscribe to the facility")
        elif predict == 1:
            print('success1')
            return render_template('UCI.html',prediction_text="Customer will subscribe to the facility")
        else:
            print('success2')
            return render_template('UCI.html',prediction_text="there is some problem in input")
        #return processed_input
    else:
        return render_template('UCI.html')
        

if __name__== "__main__":
    app.run(host='0.0.0.0',port = 8080)

