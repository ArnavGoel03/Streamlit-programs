import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model. 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Creating a Logistic Regression model. 
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating a Random Forest Classifier model.
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

@st.cache()
def prediction(model,sepal_length,petal_length,sepal_width,petal_width) :
	pred=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
	if pred[0]==0 :
		return 'iris-setosa'
	elif pred[0]==1 :
	  return 'iris-verginica' 
	elif pred[0]==2 :
		return 'Iris-versicolor'


st.sidebar.title('Iris-species prediction app')
sb1=st.sidebar.slider('Sepal-Length',float(iris_df['SepalLengthCm'].min()),float(iris_df['SepalLengthCm'].max()))
sb2=st.sidebar.slider('Sepal width',float(iris_df['SepalWidthCm'].min()),float(iris_df['SepalWidthCm'].max()))
sb3=st.sidebar.slider('Petal length',float(iris_df['PetalLengthCm'].min()),float(iris_df['PetalLengthCm'].max()))
sb4=st.sidebar.slider('Petal Width',float(iris_df['PetalWidthCm'].min()),float(iris_df['PetalWidthCm'].max()))
classifer=st.sidebar.selectbox('Classifier',('Support Vector Machine','Logistic Regression','Random Forest Classifier'))
if st.sidebar.button('Predict') :
	if classifer=='Support Vector Machine' :
		final_pred=prediction(svc_model,sb1,sb3,sb2,sb4)
		score=svc_model.score(X_train,y_train)
	elif classifer=='Logistic Regression' :
		final_pred=prediction(log_reg,sb1,sb3,sb2,sb4)
		score=log_reg.score(X_train,y_train)
	elif classifer=='Random Forest Classifier' :
		final_pred=prediction(rf_clf,sb1,sb3,sb2,sb4)
		score=rf_clf.score(X_train,y_train)
	st.write('Predicted Class :',final_pred)
	st.write('Acuuracy of the model :',score)
