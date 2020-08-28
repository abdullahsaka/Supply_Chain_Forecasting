# Import packages
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Training and test set creation
def import_data():
    data=pd.read_csv('~/Documents/Demand Forecasting/norway_car_sales.csv')
    data['Period']=data['Year'].astype(str) + "-" +data['Month'].astype(str)
    data['Period']=pd.to_datetime(data['Period']).dt.strftime("%Y-%m")
    df=pd.pivot_table(data=data,values='Quantity',index='Make',columns='Period',aggfunc='sum',fill_value=0)
    return df

df=import_data()

# Splitting data into train,test and holdout sets
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

X_train,Y_train,X_holdout,Y_holdout, X_test,Y_test=datasets(df,holdout=12)
##------------------------------------------------------------------------------------------------------------------------------
# Adaptive Boosting
ada=AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),n_estimators=100,learning_rate=0.01)
ada=ada.fit(X_train,Y_train)

Y_train_pred=ada.predict(X_train)
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
print('Ada on training set MAE:%',round(MAE_train*100,1))

Y_test_pred=ada.predict(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print('Ada on test set MAE:%',round(MAE_test*100,1))

# Parameter Optimization
from sklearn.model_selection import RandomizedSearchCV

X_train,Y_train,X_holdout,Y_holdout,X_test,Y_test = datasets(df,x_len=12,holdout=12)

n_estimators=[60,80,100,120,140]
learning_rate=[0.0001,0.0005,0.001,0.005,0.01]
param_dist={'n_estimators':n_estimators,'learning_rate':learning_rate}
results=[]

for max_depth in range(6,18,2):
    
    ada=AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth))
    ada_cv=RandomizedSearchCV(ada,param_dist,n_jobs=-1,cv=5,n_iter=20,scoring='neg_mean_absolute_error')
    ada_cv.fit(X_train,Y_train)
    
    Y_train_pred=ada_cv.predict(X_train)
    Y_holdout_pred=ada_cv.predict(X_holdout)
    Y_test_pred=ada_cv.predict(X_test)
    
    print('Tuned AdaBoost Parameters:',ada_cv.best_params_)
    result_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
    result_hold=np.mean(abs(Y_holdout-Y_holdout_pred))/np.mean(Y_holdout)
    result_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
    
    results.append([result_train,result_hold,result_test,max_depth,ada_cv.best_params_])
    
results=pd.DataFrame(results)
results.columns=['MAE Train','MAE Holdout','MAE Test','Max Depth','Best Params']
best_results=results['MAE Holdout'].idxmin()
print(results.iloc[best_results])

# New Approach: use all historical period to populate training set
X_train,Y_train,X_holdout,Y_holdout,X_test,Y_test = datasets(df,x_len=12,holdout=0)

ada=AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=100,learning_rate=0.0005)
ada.fit(X_train,Y_train)

Y_train_pred=ada.predict(X_train)
Y_test_pred=ada.predict(X_test)

MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)

print('Ada on training set MAE%:',round(MAE_train*100,1))
print('Ada on test set MAE%:',round(MAE_test*100,1))
##------------------------------------------------------------------------------------------------------------------------------
# Boosting Function
from sklearn.multioutput import MultiOutputRegressor
def AdaBoost_multi(X_train,Y_train,X_holdout,Y_holdout,X_test,Y_test) :
    base_estimator=DecisionTreeRegressor(max_depth=6)
    ada=AdaBoostRegressor(base_estimator,n_estimators=100,learning_rate=0.025)
    
    multi=MultiOutputRegressor(ada,n_jobs=-1)
    multi.fit(X_train,Y_train)
    
    Y_train_pred=multi.predict(X_train)
    Y_holdout_pred=multi.predict(X_holdout)
    Y_test_pred=multi.predict(X_test)
    
    return Y_train_pred,Y_holdout_pred,Y_test_pred

X_train,Y_train,X_holdout,Y_holdout,X_test,Y_test=datasets(df,y_len=4,x_len=12,holdout=12)

Y_train_pred,Y_holdout_pred,Y_test_pred=AdaBoost_multi(X_train,Y_train,X_holdout,Y_holdout,X_test,Y_test)

MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
MAE_holdout=np.mean(abs(Y_holdout-Y_holdout_pred))/np.mean(Y_test)

print('Ada on training set MAE%:',round(MAE_train*100,1))
print('Ada on test set MAE%:',round(MAE_test*100,1))
print('Ada on holdout set MAE%:',round(MAE_holdout*100,1))

## Extreme Gradient Boosting
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

XGB=XGBRegressor(n_jobs=-1,max_depth=10,n_estimators=100,learning_rate=0.2)
XGB=XGB.fit(X_train,Y_train)
xgb_plot=xgb.plot_importance(XGB)

# Early Stopping
x_train,x_eval,y_train,y_eval=train_test_split(X_train,Y_train,test_size=0.15)
XGB=XGBRegressor(n_jobs=-1,max_depth=10,n_estimators=1000,learning_rate=0.2)
XGB=XGB.fit(x_train,y_train,early_stopping_rounds=10,verbose=True,eval_set=[(x_eval,y_eval)],eval_metric='mae')

# Parameter Optimization: Boosting with Randomized Search
params={'max_depth':[4,5,6,7,8,10,12],
        'learning_rate':[0.001,0.05,0.01,0.025,0.05,0.1],
        'colsample_bylevel':[0.3,0.4,0.6,0.7,0.8,0.9],
        'subsample':[0.2,0.3,0.4,0.5,0.6,0.7],
        'n_estimators':[1000]}

fit_params={'early_stopping_rounds':5,
            'eval_set':[(X_holdout,Y_holdout)],
            'eval_metric':'mae',
            'verbose':False}

XGB=XGBRegressor()
XGB_cv=RandomizedSearchCV(XGB,params,cv=5,n_jobs=-1,verbose=1,n_iter=1000,scoring='neg_mean_absolute_error')
XGB_cv.fit(X_train,Y_train,**fit_params)

# Run a new model with full training set
X_train,Y_train,X_holdout,Y_holdout,X_test,Y_test=datasets(df,holdout=0)

x_train,x_eval,y_train,y_eval=train_test_split(X_train,Y_train,test_size=0.15)

XGB=XGBRegressor(n_jobs=-1,max_depth=8,n_estimators=1000,learning_rate=0.01,subsample=0.3,colsample_bylevel= 0.5)
XGB=XGB.fit(x_train,y_train,early_stopping_rounds=10,verbose=False,eval_set=[(x_eval,y_eval)],eval_metric='mae')

Y_train_pred=XGB_cv.fit(X_train)
MAE_train=np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
print('XGBoost on training set MAE%:',round(MAE_train*100,1))

Y_test_pred=XGB_cv.fit(X_test)
MAE_test=np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
print('XGBoost on test set MAE%:',round(MAE_test*100,1))

# Forecast multiple periods
def XGBoost(X_train,Y_train,X_test,params):
    from sklearn.model_selection import train_test_split
    x_train,x_eval,y_train,y_eval=train_test_split(X_train,Y_train,test_size=0.15)
    
    fit_params={'early_stopping_rounds':5,
            'eval_set':[(x_eval,y_eval)],
            'eval_metric':'mae',
            'verbose':False}
    
    XGB=XGBRegressor(**params)
    XGB=XGB.fit(x_train,y_train,**fit_params)
    
    return XGB.predict(X_train),XGB.predict(X_test)

def XGBoost_multi(X_train,Y_train,X_test,params):
    
    Y_train_pred=Y_train.copy()
    Y_test_pred=Y_test.copy()
    
    for col in range(Y_train.shape[1]):
        results=XGBoost(X_train,Y_train[:,col],X_test,params)
        Y_test_pred[:,col]=results[0]
        Y_test_pred[:,col]=results[1]
    
    return Y_test_pred,Y_train_pred
    





