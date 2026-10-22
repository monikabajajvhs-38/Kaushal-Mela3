import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data=pd.read.csv('diabetes.csv')
x=data.drop('Outcome', axis=1)
y=data['Outcome']
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict the likelihood of diabetes.")
input_data=[]
for col in x.columns:
  value=st.number_input(f"Enter{col}", min_value=0.0, step=0.1)
  input_data.append(value)
  if st.button("Predict"):
    prediction=model.predict([input_data])
    result="Diabetic" if prediction[0]==1 else "Non-diabetic"
    st.success(f"Prediction: {result}")
st.subheader("Model performance")
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification report: ")
st.text(classification_report(y_test, y_pred))
