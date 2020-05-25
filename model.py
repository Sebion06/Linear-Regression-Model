"""
Linear Regression Model for predicting restaurant prices using the scikit-learn Linear Regression Algorithm

Copyright 2020 Moticica Sebastian-Ionut, Deaconu Constantin

"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#Read the data
training_df = pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

#Fill NaN values if any
training_df.fillna(0)
test_df.fillna(0)

pd.set_option('display.max_columns', None)

#scale the label into units of MILLION$
training_df["revenue"] /= 1000000.0

#Get header list for all values (just in case)
items_list=[]
'''
for items in training_df.head():
    if items[0]=='P':
        items_list.append(items)
items_list=['P2','P28']
'''

#Show heatmap of relevant features
plt.figure(figsize=(24,22))
cor = training_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Select correlated features
cor_target = abs(cor["revenue"])
relevant_features = cor_target[cor_target>0.11]    #get features that have some correlation with the target value
items_list=list(relevant_features.index.values)
del items_list[-1]  #remove the last element ("revenue") as the correlation is 1.0

#Set the data
x=training_df[items_list].values
y=training_df['revenue'].values

#Check out an estimate of the "revenue" column and plot it
plt.figure(figsize=(16,8))
plt.tight_layout()
sns.distplot(training_df['revenue'], label="training revenue estimate")
plt.legend()
plt.show()

#Separate the train data into training and validation data (85% to 15%)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=0)

# Create the model and fit it
model= LinearRegression()
model.fit(x_train,y_train)


print("Model features: "+str(items_list))
print("Model coefficients:" +str(model.coef_))
print("Model intercept: "+ str(model.intercept_))

#Make predictions on validation data
y_val_pred = model.predict(x_val)

#Check out the score and the mean squared error
print('Mean Squared Error:', mean_squared_error(y_val, y_val_pred))

#View how close the predictions were to the actual data (if wanted)
''' 
df = pd.DataFrame({'Actual value': y_val, 'Predicted value': y_val_pred})
df1 = df.head(25)
print(df1)
'''

#Plot validation data and predicted validation data
fig, ax = plt.subplots()
ax.scatter(y_val, y_val_pred)
ax.plot([y_val.min(), y_val_pred.max()], [y_val.min(), y_val_pred.max()], 'k--', lw=4)
ax.set_xlabel('Actual validation data')
ax.set_ylabel('Predicted validation data')
plt.show()

#Make predictions on the test data
x_test=test_df[items_list].values
y_test = model.predict(x_test)

#Plot y_test values
plt.figure(figsize=(16,8))
plt.tight_layout()
sns.distplot(y_test, label='Test revenue estimate')
plt.legend()
plt.show()

#Write data to file
np.savetxt("results.csv", np.dstack((np.arange(1, y_test.size+1),y_test))[0],"%d,%.3f",header="Id,Values")