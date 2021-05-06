import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve
from lightgbm import LGBMClassifier,plot_importance
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test1.csv")

#transformasi kategori
train.state = pd.Categorical(train.state).codes
train.area_code = pd.Categorical(train.area_code).codes
train.international_plan = pd.Categorical(train.international_plan).codes
train.voice_mail_plan = pd.Categorical(train.voice_mail_plan).codes
train.churn = pd.Categorical(train.churn).codes
test.state = pd.Categorical(test.state).codes
test.area_code = pd.Categorical(test.area_code).codes
test.international_plan = pd.Categorical(test.international_plan).codes
test.voice_mail_plan = pd.Categorical(test.voice_mail_plan).codes

#pembagian data
X = train.drop('churn',axis=1)
y  = train.churn
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

X_train = train.drop('churn',axis=1)
y_train = train.churn
X_test = test.drop('id',axis=1)

light = LGBMClassifier(n_estimators=200,learning_rate=0.11,
                      min_child_samples=30,num_leaves=60)
light.fit(X_train,y_train)

pred = light.predict(X_test)

plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(light,color='navy',)

