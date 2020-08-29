#importing packages
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

#load the CSV file(should be in the same directory)
car_sales=pd.read_csv('~/Documents/Demand Forecasting/norway_car_sales.csv')
car_sales.head()

#create a column 'Period' with both the Year and the Month
car_sales['Period']=car_sales['Year'].astype(str) + "-" + car_sales['Month'].astype(str)

#we use the datetime formatting to make sure format is consistent
car_sales['Period']=pd.to_datetime(car_sales['Period']).dt.strftime('%Y-%m')

#create a pivot of the data to show the periods on columns and the car makers on rows
df=pd.pivot_table(data=car_sales,values='Quantity',index='Make',columns='Period',aggfunc='sum',fill_value=0)

#print data to Excel for reference
df.to_excel('Clean Demand.xlsx')

#Splitting data into train and test sets
def datasets(df,x_len=12,y_len=1,y_test_len=12):
    D=df.values
    periods=D.shape[1]
    
    #training set creation:run through all the possible time windows
    loops=periods+1-x_len-y_len-y_test_len 
    train=[]
    
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train=np.vstack(train)
    X_train,Y_train=np.split(train,[x_len],axis=1)
    
    #Test set creation:unseen 'future' data with the demand just before
    max_col_test=periods-x_len-y_len+1
    test=[]
    for col in range(loops,max_col_test):
        test.append(D[:,col:col+x_len+y_len])
    test=np.vstack(test)
    X_test,Y_test=np.split(test,[x_len],axis=1)

    #this data formatting is needed if we only predict a single period
    if y_len==1:
        Y_train=Y_train.ravel()
        Y_test=Y_test.ravel()
        
    return X_train,X_test,Y_train,Y_test

X_train,X_test,Y_train,Y_test=datasets(df)

# 1.Linear Regression Model
#create a linear regression model
reg=LinearRegression()
reg=reg.fit(X_train,Y_train) #fit it to the training data

#create two predictions for the training and test sets
Y_train_pred= reg.predict(X_train)
Y_test_pred= reg.predict(X_test)

#compute MAE for both the training and test sets
Y_train_pred=reg.predict(X_train)
Y_test_pred=reg.predict(X_test)

#compute MAE for both the training and test sets
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)

#print the results
print("Linear Regression Train MAE%",round(MAE_train*100,1))
print("Linear Regression Test MAE%",round(MAE_test*100,1))

# 2.Regression Trees
#instantiate a Decision Tree Regressor
tree=DecisionTreeRegressor(max_depth=5,min_samples_leaf=5)

#fit the tree to the training data
tree.fit(X_train,Y_train)

#create a prediction based on our model
Y_train_pred=tree.predict(X_train)

#compute the Mean Absolute Error of the model
MAE_tree=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_test)

#print the results
print('Regression Tree on the train set MAE%',round(MAE_tree*100,1))

Y_test_pred=tree.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print('Regression Tree on test set MAE%',round(MAE_test*100,1))

for criterion in ['mse','mae']:
    start_time=time.time()
    tree=DecisionTreeRegressor(max_depth=5,min_samples_leaf=5,criterion=criterion)
    tree.fit(X_train,Y_train)
    Y_test_pred=tree.predict(X_test)
    MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
    print(criterion)
    print("%s seconds" % round(time.time()-start_time,2))
    print("MAE%",round(MAE_test*100,2))
    print()

# Parameter Optimization on Regression Trees 
max_depth=list(range(5,11))
max_depth.append(None)
min_samples_leaf=range(5,20)
param_dist={"max_depth":max_depth,"min_samples_leaf":min_samples_leaf}

tree=DecisionTreeRegressor()
tree_cv=RandomizedSearchCV(tree,param_dist,n_jobs=-1,cv=10,verbose=1,n_iter=100,scoring="neg_mean_absolute_error")
tree_cv.fit(X_train,Y_train)
print("Tuned Regression Tree Parameters",tree_cv.best_params_)

Y_train_pred=tree_cv.predict(X_train)
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
print("Tree on training set MAE%",round(MAE_train*100,1))

Y_test_pred=tree_cv.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print("Tree on test set MAE%",round(MAE_test*100,1))

#Random Forest Model
forest=RandomForestRegressor(bootstrap=True,max_features='auto',min_samples_leaf=18,max_depth=7)
forest.fit(X_train,Y_train)
Y_test_pred=forest.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print("Random Forest on test set MAE%",round(MAE_test*100,1))

# Parameter Optimization on Random Forest
max_features=range(3,8)
max_depth=range(6,11)
min_samples_split=range(5,15)
min_samples_leaf=range(5,15)
bootstrap=[True,False]
param_dist={'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'bootstrap':bootstrap}

forest=RandomForestRegressor(n_estimators=50,n_jobs=1)
forest_cv=RandomizedSearchCV(forest,param_dist,cv=6,n_jobs=-1,
                             verbose=2,n_iter=400,scoring="neg_mean_absolute_error")
forest_cv.fit(X_train,Y_train)

print('Tuned Forest Parameters:',forest_cv.best_params_)

Y_train_pred=forest_cv.predict(X_train)
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
print('Random Forest on training set MAE%',round(MAE_train*100,1))

Y_test_pred=forest_cv.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print("Random Forest on test set MAE%",round(MAE_test*100,1))

# Random Forest with best parameters found above
forest=RandomForestRegressor(n_estimators=200,min_samples_split=10,
                             min_samples_leaf=7,max_features=4,
                             max_depth=9,bootstrap=True)

forest.fit(X_train,Y_train)

Y_train_pred=forest.predict(X_train)
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
print('Random Forest on training set MAE%',round(MAE_train*100,1))

Y_test_pred=forest.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print("Random Forest on test set MAE%",round(MAE_test*100,1))

# Feature Importance
features=[]
columns=X_train.shape[1]
for column in range(columns):
    features.append('M--'+str(columns-column))
    
imp_forest=forest.feature_importances_.reshape(-1,1)
importances=pd.DataFrame(imp_forest,index=features,columns=['Forest'])
importances.plot(kind='bar')








    





