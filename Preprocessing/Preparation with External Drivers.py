import pandas as pd
import numpy as np

# Training and test set creation
def import_data():
    data=pd.read_csv('~/Documents/Demand Forecasting/norway_car_sales.csv')
    data['Period']=data['Year'].astype(str) + "-" +data['Month'].astype(str)
    data['Period']=pd.to_datetime(data['Period']).dt.strftime("%Y-%m")
    df=pd.pivot_table(data=data,values='Quantity',index='Make',columns='Period',aggfunc='sum',fill_value=0)
    return df
df=import_data()

gdp_data=pd.read_csv('~/Desktop/GDP.csv').set_index('Year')

dates=[int(date[:4]) for date in df.columns.values]

X_exo=[gdp_data.loc[date,'GDP'] for date in dates]

def datasets(df,X_exo,x_len=12,y_len=1,y_test_len=12):
    periods=df.shape[1]
    D=df.values
    X_exo=np.array(X_exo).reshape([1,-1])
    X_exo=np.repeat(X_exo,D.shape[0],axis=0)
    
    #training set creation:run through all the possible time windows
    loops=periods+1-x_len-y_len-y_test_len 
    train=[]
    
    for col in range(loops):
        d=D[:,col:col+x_len+y_len]
        exo=X_exo[:,col:col+x_len+y_len]
        train.append(np.hstack([exo,d]))
    train=np.vstack(train)
    X_train,Y_train=np.split(train,[-y_len],axis=1)
    
    #Test set creation:unseen 'future' data with the demand just before
    max_col_test=periods-x_len-y_len+1
    test=[]
    for col in range(loops,max_col_test):
        d=D[:,col:col+x_len+y_len]
        exo=X_exo[:,col:col+x_len+y_len]
        test.append(np.hstack([exo,d]))
    test=np.vstack(test)
    X_test,Y_test=np.split(test,[-y_len],axis=1)

    #this data formatting is needed if we only predict a single period
    if y_len==1:
        Y_train=Y_train.ravel()
        Y_test=Y_test.ravel()
        
    return X_train,X_test,Y_train,Y_test

X_train,X_test,Y_train,Y_test=datasets(df,X_exo)
