# Load required packages
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# Training and test set creation
def import_data():
    data=pd.read_csv('./Documents/Demand Forecasting/norway_car_sales.csv')
    data['Period']=data['Year'].astype(str) + "-" +data['Month'].astype(str)
    data['Period']=pd.to_datetime(data['Period']).dt.strftime("%Y-%m")
    df=pd.pivot_table(data=data,values='Quantity',index='Make',columns='Period',aggfunc='sum',fill_value=0)
    return df

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
        
    return X_train,Y_train,X_test,Y_test

df=import_data()
X_train,Y_train,X_test,Y_test=datasets(df)
##------------------------------------------------------------------------------------------------------------------------------
# Extremely Randomized Trees
ETR= ExtraTreesRegressor(n_estimators=200,min_samples_split=10,min_samples_leaf=7,max_features=6,max_depth=9,bootstrap=True)
ETR.fit(X_train,Y_train)

Y_train_pred=ETR.predict(X_train)
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
print("ETR on training set MAE%:",round(MAE_train,2))

Y_test_pred=ETR.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print("ETR on test set MAE%:",round(MAE_test,2))

# Tuned Optimization: Extremely Randomized Trees
max_features=range(6,11)
max_depth=range(8,15)
min_samples_split=range(2,10,2)
min_samples_leaf=range(2,10,2)
bootstrap=[True,False]

param_dist={'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'bootstrap':bootstrap}

ETR=ExtraTreesRegressor(n_estimators=200,n_jobs=1)
ETR_cv=RandomizedSearchCV(ETR,param_dist,cv=5,verbose=5,n_jobs=-1,n_iter=300,scoring='neg_mean_absolute_error')
ETR_cv.fit(X_train,Y_train)

print('Tuned ETR Parameters:',ETR_cv.best_params_)

Y_train_pred=ETR_cv.predict(X_train)
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
print('ETR on training set MAE%',round(MAE_train*100,2))

Y_test_pred=ETR_cv.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print('ETR on test set MAE%',round(MAE_test*100,2))
##------------------------------------------------------------------------------------------------------------------------------
# Feature Initializaton
forest_features={'n_jobs':-1,
                 'n_estimators':100,
                 'max_features':0.3,
                 'bootstrap':False,
                 'max_depth':9,
                 'min_samples_split':12,
                 'min_samples_leaf':8}

ETR_features={'n_jobs':-1,
                 'n_estimators':100,
                 'max_features':0.9,
                 'bootstrap':False,
                 'max_depth':10,
                 'min_samples_split':7,
                 'min_samples_leaf':4}

n_months=range(6,50,2)
results=[]
# How to otimize how many previous periods we should take into account to make a prediction??
# Feature Optimization: First Experiment
for x_len in n_months: #we loop through the different x_len
    X_train,Y_train,X_test,Y_test=datasets(df,x_len=x_len)
    
    forest=RandomForestRegressor(**forest_features)
    ETR=ExtraTreesRegressor(**ETR_features)
    models=[('Forest',forest),('ETR',ETR)]
    
    for name,model in models: # we loop through the models
        model.fit(X_train,Y_train)
        Y_train_pred=model.predict(X_train)
        mae_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
        
        Y_test_pred=model.predict(X_test)
        mae_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
        
        results.append([name + 'Train',mae_train,x_len])
        results.append([name + 'Test',mae_test,x_len])
        
data=pd.DataFrame(results,columns=['Model','MAE%','Number of Months'])

data=data.set_index(['Number of Months','Model']).stack().unstack('Model')

data.index=data.index.droplevel(level=1)
data.index.name='Number of months'

data.plot(color=['orange']*2+['black']*2,style=['-','--']*2)
##------------------------------------------------------------------------------------------------------------------------------
# Feature Optimization: Second Experiment
for x_len in n_months:
     X_train,Y_train,X_test,Y_test= datasets(df,x_len=x_len)
     
     forest=RandomForestRegressor(**forest_features)
     ETR=ExtraTreesRegressor(**ETR_features)
     models=[('Forest',forest),('ETR',ETR)]
     
     kf=KFold(n_splits=8)
     
     for name,model in models:
         
         mae_kfold_train=[]
         mae_kfold_val=[]
         
         for train_index,val_index in kf.split(X_train):
         
             X_train_kfold,X_val_kfold=X_train[train_index],X_train[val_index]
             Y_train_kfold,Y_val_kfold=Y_train[train_index],Y_train[val_index]
         
             model.fit(X_train_kfold,Y_train_kfold)
             Y_train_pred=model.predict(X_train_kfold)
             mae_kfold_train.append(np.mean(abs(Y_train_kfold-Y_train_pred))/np.mean(Y_train_kfold))
         
             Y_val_pred=model.predict(X_val_kfold)
             mae_kfold_val.append(np.mean(abs(Y_val_kfold-Y_val_pred))/np.mean(Y_val_kfold))
         
         model.fit(X_train,Y_train)
         Y_test_pred=model.predict(X_test)
         mae_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
    
         results.append([name + 'Val',np.mean(mae_kfold_val),x_len])
         results.append([name + 'Train',np.mean(mae_kfold_train),x_len])
         results.append([name + 'Test',np.mean(mae_test),x_len])

data=pd.DataFrame(results,columns=['Model','MAE%','Number of Months'])

data=data.set_index(['Number of Months','Model']).stack().unstack('Model')

data.index=data.index.droplevel(level=1)
data.index.name='Number of months'

print(data.idxmin())

data.plot(color='orange'*3+['black']*3,style=['-','--']*2)    
##------------------------------------------------------------------------------------------------------------------------------
# Feature Optimization: Third Experiment
def datasets(df,x_len=12,y_len=1,y_test_len=12,holdout=0):
    D=df.values
    periods=D.shape[1]
    
    #training set creation: run through all the possible time windows
    loops=periods + 1 - x_len - y_len - y_test_len
    train=[]
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train=np.vstack(train)
    X_train,Y_train= np.split(train,[x_len],axis=1)
    
    rows=df.shape[0]
    if holdout>0 :
        X_train,X_holdout=np.split(X_train,[-rows*holdout],axis=0)
        Y_train,Y_holdout=np.split(Y_train,[-rows*holdout],axis=0)
    else:
        X_holdout=np.array([])
        Y_holdout=np.array([])
    
    #test set creation:unseen 'future' data with the demand just before
    max_col_test=periods -x_len -y_len +1
    test=[]
    for col in range(loops,max_col_test):
        test.append(D[:,col:col+x_len+y_len])
    test=np.vstack(test)
    X_test,Y_test=np.split(test,[x_len],axis=1)
    
    #this data formatting is needed if we only predict a single period
    if y_len==1:
        Y_train=Y_train.ravel()
        Y_test=Y_test.ravel()
        Y_holdout=Y_holdout.ravel()
    
    return X_train,Y_train,X_holdout,Y_holdout,X_test,Y_test

for x_len in n_months: #we loop through the different x_len
    X_train,Y_train,X_holdout,Y_holdout, X_test,Y_test=datasets(df,x_len=x_len,holdout=12)
    
    forest=RandomForestRegressor(**forest_features)
    ETR=ExtraTreesRegressor(**ETR_features)
    models=[('Forest',forest),('ETR',ETR)]
    
    for name,model in models: # we loop through the models
        model.fit(X_train,Y_train)
        Y_train_pred=model.predict(X_train)
        mae_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
        
        Y_test_pred=model.predict(X_test)
        mae_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
        
        Y_holdout_pred=model.predict(X_holdout)
        mae_holdout=np.mean(abs(Y_holdout-Y_holdout_pred))/np.mean(Y_holdout)
        
        results.append([name + 'Train',mae_train,x_len])
        results.append([name + 'Test',mae_test,x_len])
        results.append([name + 'Holdout',mae_holdout,x_len])
        
data=pd.DataFrame(results,columns=['Model','MAE%','Number of Months'])

data=data.set_index(['Number of Months','Model']).stack().unstack('Model')

data.index=data.index.droplevel(level=1)
data.index.name='Number of months'

print(data.idxmin())

data.plot(color=['orange']*3+['black']*3,style=['-','--',':']*3)