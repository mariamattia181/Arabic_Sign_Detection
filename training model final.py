#%%
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences

#%%
#Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
print(data_dict.keys())

#%%

data = data_dict['data']
max_sequence_length = 42  
padded_data = pad_sequences(data, maxlen=max_sequence_length, padding='post', truncating='post', dtype='float32')
padded_data_array = np.array(padded_data)
labels = np.array(data_dict['labels'])



#%%
# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(padded_data_array, labels, stratify=labels, test_size=0.2, shuffle=True)


#%%
#SVC PART

svm_model = SVC()  
svm_model.fit(x_train, y_train)
y_predict = svm_model.predict(x_test)

#%%
# Calculating accuracy
score = accuracy_score(y_predict, y_test)
print('Accuracy equals:{}'.format(score * 100))

#%%
from sklearn.model_selection import GridSearchCV

param_grid=[
    {'C':[0.5,1,10,100],
     'gamma':['scale',1,0.1,0.001,0.00001],
     'kernel':['rbf'],
            }
           ]

optional_params=GridSearchCV(SVC(),param_grid,cv=5,scoring='accuracy',verbose=0)
optional_params.fit(x_train,y_train)
print(optional_params.best_params_)

#%%
svm_model = SVC(C=10,gamma='scale',kernel='rbf')  
svm_model.fit(x_train, y_train)
y_predict = svm_model.predict(x_test)

# Calculating accuracy
score = accuracy_score(y_predict, y_test)
print('Accuracy equals:{}'.format(score * 100))


#%%

#Heatmap svc
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


conf_matrix = confusion_matrix(y_test, y_predict)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


print(classification_report(y_test, y_predict))
#%%
#CROSS VAL lel SVM
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=6, shuffle=True, random_state=5)
cv_scores = cross_val_score(svm_model, padded_data_array, labels, cv=kf)


print(cv_scores)

#%%

print(np.mean(cv_scores))
print(np.std(cv_scores))
print(np.quantile(cv_scores, [0.025, 0.975]))

#%%

f=open('modelSVC.pickle','wb')
pickle.dump({'modelSVC':svm_model},f)
f.close()  

#%%

#random forest classifier
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

#%%
# Calculating accuracy
score = accuracy_score(y_predict, y_test)
print('Accuracy equals {}'.format(score * 100))




#%%

#Heatmap randomforest

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


conf_matrix = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_predict))

#%%
#CROSS VAL lel RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=6, shuffle=True, random_state=5)

cv_scores = cross_val_score(model, padded_data_array, labels, cv=kf)

print(cv_scores)

#%%


print(np.mean(cv_scores))
print(np.std(cv_scores))
print(np.quantile(cv_scores, [0.025, 0.975]))

#%%
f=open('modelRF.pickle','wb')
pickle.dump({'model':model},f)
f.close()



#%%
#xgbboost
import pickle
from keras.utils import pad_sequences
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder

#%%
le = LabelEncoder()
Labels_Encoded = le.fit_transform(labels)

#%%
X_Train, X_Test, Y_Train, Y_Test = train_test_split(padded_data_array, Labels_Encoded, test_size=0.2, shuffle=True, stratify=labels)

#%%
XGBoostClassModel = XGBClassifier(n_estimators=350, gamma=0.01,learning_rate=0.1)
XGBoostClassModel.fit(X_Train, Y_Train)

Y_Predict = XGBoostClassModel.predict(X_Test)
Predictions = [round(value) for value in Y_Predict]

#%%
Score = accuracy_score(Y_Test,Predictions)

print('Accuracy equals: {}'.format(Score * 100))

#%%

#CROSS VAL lel XGBoostClassifier
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=6, shuffle=True, random_state=5)

cv_scores = cross_val_score(XGBoostClassModel, padded_data_array, Labels_Encoded, cv=kf)

print(cv_scores)


#%%

print(np.mean(cv_scores))

print(np.std(cv_scores))

print(np.quantile(cv_scores, [0.025, 0.975]))

#%%

file = open('XGBoostClassModel.p', 'wb')
pickle.dump({'XGBoostClassModel': XGBoostClassModel}, file)
file.close()

