# Load packages
import pandas as pd
import numpy as np

# Training and test set creation
def import_data():
    data=pd.read_csv('~/Documents/Demand Forecasting/norway_car_sales.csv')
    data['Period']=data['Year'].astype(str) + "-" +data['Month'].astype(str)
    data['Period']=pd.to_datetime(data['Period']).dt.strftime("%Y-%m")
    df=pd.pivot_table(data=data,values='Quantity',index='Make',columns='Period',aggfunc='sum',fill_value=0)
    return df

df=import_data() # execute the function

# Data Preparation
luxury_brands=['Aston Martin','Bentley','Ferrari','Jaguar','Lamborghini','Lexus','Lotus','Maserati','McLaren','Porsche','Tesla']

df['Segment']=[brand in luxury_brands for brand in df.index]

df.Segment.replace({True:'Luxury',False:'Normal'},inplace=True)

df=pd.get_dummies(df,columns=['Segment'],prefix_sep='_')

#Splitting data into train and test sets
def datasets(df,x_len=12,y_len=1,y_test_len=12,sep='_'):
    col_cat=[col for col in df.columns if sep in col]
    
    D=df.drop(col_cat,axis=1).values
    periods=D.shape[1]
    C=df[col_cat].values
    
    #training set creation:run through all the possible time windows
    loops=periods+1-x_len-y_len-y_test_len 
    train=[]
    
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train=np.vstack(train)
    X_train,Y_train=np.split(train,[x_len],axis=1)
    X_train=np.hstack((np.vstack([C]*loops),X_train))
    
    #Test set creation:unseen 'future' data with the demand just before
    max_col_test=periods-x_len-y_len+1
    test=[]
    for col in range(loops,max_col_test):
        test.append(D[:,col:col+x_len+y_len])
    test=np.vstack(test)
    X_test,Y_test=np.split(test,[x_len],axis=1)
    X_test=np.hstack((np.vstack([C]*(max_col_test-loops)),X_test))

    #this data formatting is needed if we only predict a single period
    if y_len==1:
        Y_train=Y_train.ravel()
        Y_test=Y_test.ravel()
        
    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test=datasets(df)

# Integer Coding

df=import_data() # execute the function

df['Segment']=[brand in luxury_brands for brand in df.index]

df.Segment.replace({True:'Luxury',False:'Normal'},inplace=True)
df.Segment.replace({'Normal':1,'Luxury':2},inplace=True)

X_train,Y_train,X_test,Y_test=datasets(df,x_len=12,y_len=1,y_test_len=12,sep='Segment')






