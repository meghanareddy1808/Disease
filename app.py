import streamlit as st
rad=st.sidebar.radio("Navigation Menu",["Home","Diabetes"])
if rad=="Home":
    st.title("Diabetes Predictions App")
    st.image("medical.jpg")
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Diabetes Prediction

#loading the Diabetes dataset
df2=pd.read_csv("diabetes.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x2) & target(y2)
x2=df2.iloc[:,[1,4,5,7]].values
x2=np.array(x2)
y2=y2=df2.iloc[:,[-1]].values
y2=np.array(y2)
#performing train-test split on the data
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model2=RandomForestClassifier()
#fitting the model with train data (x2_train & y2_train)
model2.fit(x2_train,y2_train)

#Diabetes Page

#heading over to the Diabetes section
if rad=="Diabetes":
    st.header("Know If You Are Affected By Diabetes")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Glucose (glucose), Insulin (insulin), Body Mass Index-BMI (bmi), Age (age)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    glucose=st.number_input("Enter Your Glucose Level (0-200)",min_value=0,max_value=200,step=1)
    insulin=st.number_input("Enter Your Insulin Level In Body (0-850)",min_value=0,max_value=850,step=1)
    bmi=st.number_input("Enter Your Body Mass Index/BMI Value (0-70)",min_value=0,max_value=70,step=1)
    age=st.number_input("Enter Your Age (20-80)",min_value=20,max_value=80,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Result"):
        if prediction2==1:
            st.warning("Might Be Affected By Diabetes")
        elif prediction2==0:
            st.success("Safe")
