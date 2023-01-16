import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pycaret.classification import *

from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import pickle

ickled_model = pickle.load(open('../Model/modelo_xgboost_09_AUC.pkl', 'rb'))

df = pd.read_excel('../BD/NewOpportunitiesList.xlsx')

X_numerical = df.drop(['Supplies Subgroup', 'Opportunity Number', 'Supplies Group', 'Competitor Type','Lon', 'Lat', 'Client Size By Revenue',
 'Client Size By Employee Count', 'Country', 'ID'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X_numerical)
X = pd.DataFrame(X)

pred = pickled_model.predict(X)

pred_p = pickled_model.predict_proba(X)

df['Probabilidad'] = pred_p[:,1]

df.to_excel('../BD/Resultados_Prediccion.xlsx')

df.to_csv('../BD/Resultados_Prediccion.csv', sep=';', index=False)
