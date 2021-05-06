import numpy as np
import pandas as pd

dataLatih = pd.read_csv("data/train.csv")


#mengubah data kategori menjadi numerik
dataLatih.area_code = dataLatih.area_code.map({'area_code_415':415,'area_code_408':408,'area_code_510':510})
dataLatih = dataLatih.replace({'voice_mail_plan':{'yes':1,'no':0}})
dataLatih = dataLatih.replace({'international_plan':{'yes':1,'no':0}})
dataLatih = dataLatih.replace({'churn':{'yes':1,"no":0}})
dataLatih.state = dataLatih.state.astype('category')

#menghapus kolom yg memiliki korelasi
col_drop = ['total_day_minutes','total_night_minutes','total_eve_minutes','total_intl_minutes']
dataLatih = dataLatih.drop(columns = col_drop,axis = 1)

X = dataLatih.drop(['state','churn'],axis = 1)
y = dataLatih['churn']

#featurescalling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
df = pd.DataFrame(X_std, index = dataLatih.index, columns = dataLatih.columns[1:15])
df['state'] = dataLatih['state']
df['churn'] = dataLatih['churn']

#membagi data menajadi dat training dan data test
X_new = df.drop(['state','churn'],axis = 1)
y_new = df['churn']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_new,y_new,test_size = 0.3,random_state = 42)

#Data latih dengan Random Forest
from sklearn.ensemble import RandomForestRegressor
for_reg = RandomForestRegressor(random_state = 42)
for_reg.fit(X_train,y_train)

#menyimpan model
import joblib
joblib.dump(for_reg, 'model.pkl')

#menyimpan kolom
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')


