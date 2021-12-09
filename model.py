from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle
dengue_features = pd.read_excel("DengueData.xlsx")
dengue_labels= dengue_features["No. of cases"]
dengue_features=dengue_features.drop(labels="No. of cases",axis=1)
print(dengue_features.head())
print(dengue_labels)
dengue_features.shape[0]
dengue_features["Avg Temp.(F)"]=(dengue_features["Avg Temp.(F)"]- 32)* 5/9
dengue_features["Avg Temp.(C)"]=dengue_features["Avg Temp.(F)"]
dengue_features=dengue_features.drop(labels="Avg Temp.(F)", axis=1)
print(dengue_features)
lmap={"Jan":0,"Feb":1,"Mar":2,"Apr":3,"May":4,"Jun":5, "Jul":6, "Aug":7, "Sep":8, "Oct":9, "Nov":10, "Dec":11}
dengue_features["Month "]=dengue_features["Month "].map(lmap)
dengue_features=dengue_features.drop("City", axis=1)
print(dengue_features.head())
X=dengue_features.values
Y=dengue_labels.values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state=1)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(x_train, y_train)
pickle.dump(knn,open('model.sav','wb'))
model=pickle.load(open('model.sav','rb'))
print(model.predict(x_test))

#y_pred2 = knn.predict(x_test)
#from sklearn.metrics import mean_absolute_error
#print(mean_absolute_error(y_test, y_pred2))
#print(y_test)
#print(y_pred2)
#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(n_estimators=100)
#rf.fit(x_train, y_train)
#y_pred = rf.predict(x_test)
#print(mean_absolute_error(y_test, y_pred))
#def svm_train(X, labels, C_val):
#    """
#    SVM train
#    Parameters
#    ----------
#    hog_feature: numpy array hog features.
#                 training features, formated as [[hog1], [hog2], ...]
#
#    labels: numpy array labels (1-dim)
#            training labels
#    C_val: float
#           val of c parameter for SVM model
#    Return:
#    clf: SVM model weights
#    """
#    clf = SVR(C=C_val, tol=1e-3)
#    clf.fit(X, labels)
#    return clf
##TRAINING SVM MODEL BY CHOSING THE HYPERPARAMETER ON THE VAL SET
#
## SVM classification to predict 1 for water logged image and 0 for non waterlogged image
#
#C = [2000,3000,4000,5000]
#max_accuracy = 0
#best_clf= None
#
#for c_val in C:
#    # call svm train
#    clf=svm_train(x_train,y_train, c_val)
#    #y_pred1=clf.predict(x_train)
#    #print(mean_absolute_error(y_train, y_pred1))
#    y_pred=clf.predict(x_test)
#    print(mean_absolute_error(y_test, y_pred))
#    clf=svm_train(x_train,y_train, 4000)
#y_pred=clf.predict(x_test)
#print(mean_absolute_error(y_test, y_pred))
#print(y_pred)
#print(y_test)
#print(np.sum(np.fabs(y_pred-y_test)))
#y_pred3=0.5*(y_pred+y_pred2)
#print(mean_absolute_error(y_test, y_pred3))
#print(y_pred3)
#print(y_test)
#for i in range(0,len(y_pred3)):
#    if y_pred3[i]<0:
#        y_pred3[i]=0
#print(mean_absolute_error(y_test, y_pred3))
#print(y_pred3)
#print(y_test)
#pickle.dump(knn,open('model.pkl','wb'))
#model=pickle.load(open('model.pkl','rb'))
#print(model.predict(x_test))