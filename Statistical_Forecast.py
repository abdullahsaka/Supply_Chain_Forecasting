#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Abdullah Saka
title: Supply Chain Forecasting - 1
"""

#1. Moving Average(MA) method
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

def moving_average(d,extra_periods,n):
    #INPUTS
    #d: a time series that contains the historical demand
    #extra_periods: the number of periods we want to forecast in the future
    #n: the number of periods we will average
    #Transform the input into a numpy array
    d = np.array(d)
    #Historical period length
    cols = len(d)
    #append np.nan into the demand array to cover future periods
    d = np.append(d,[np.nan]*extra_periods)
    #define the forecast array
    f = np.full(cols + extra_periods, np.nan)
    
    #Create all the t+1 forecasts until end of historical period
    for t in range(n,cols+1):
        f[t] = np.mean(d[t-n:t])
    
    #Forecast for all extra periods
    f[cols+1:] = f[t]
    
    #Return a dataframe with the demand, forecast & error
    df = pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Error':d-f})
    
    return df

#First Python Basics

#creates a list
ts = [1,2,3,4,5,6]

#second list
ts2 = [10,20,30,40,50,60]

#we can't add two lists since this operation execute concatenation.
ts + ts2

#that is the reason we have numpy library
#this library was initially released in 2005. It offers us a new data type: a Numpy array
#similar to a list, but differs in that we can easily call any mathematical function on them

#we can create an array from a list
ts = np.array([1,2,3,4,5,6])
ts2 = np.array([10,20,30,40,50,60])

#now, we can add two arrays
ts + ts2

#we can use any mathematical function on a list using numpy
alist = [1,2,3]
np.mean(alist)

#for help on numpy, go to www.docs.scipy.org/doc/numpy. 
#most of the Google searches on numpy functions will direct us to this website

#SLICING ARRAYS/LISTS
alist = ["cat","dog","mouse"]
alist[1]

anarray = np.array([1,2,3])
anarray[0]

alist[1:] #returns a list starting with 2nd element

anarray[:1] #returns an array ending with 2nd element, but not including it

alist[-1] #returns last element in the list
alist[:-1] 

#PANDAS
#released in 2008 by Wes McKinney. Names comes from PANel DAta
test_pd = pd.DataFrame([ts,ts2])

test_pd.columns = ["Day1","Day2","Day3","Day4","Day5","Day6"]
test_pd

#For documentation, go to, www.pandas.pydata.org/pandas-docs/stable/

#SLICING DATA FRAMES

#April 11, 2020
test_pd['Day3']
test_pd.Day3
#test_pd[0]    #Not sure why this is not working. It is supposed to give the first row using index
#test_pd.index[0]
test_pd.loc[0, 'Day3']
test_pd.iloc[0,2]

#Dictionaries
#Another way to create a DataFrame is to construct it based on a dictionary of lists/arrays
dic = {'Small product' : ts, 'Big product' : ts2}
dic

#Getting value based on key
dic['Small product']
dic['Small product'] + dic['Big product']

#Create data frame directly based on dictionary
#keys in dictionary become columns in data frame, values become rows
df = pd.DataFrame.from_dict(dic)

d = [28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]


# apply moving average in the small dataset
df_ma = moving_average(d,10,6)

#Visualization with pandas
#we can plot any DataFrame simply by calling the method .plot()
df_ma[['Demand','Forecast']].plot()

#we can customize .plot() by specifying some parameters.
#   figsize(width,height) : defines the size of the figure in inches
#   title : displays a title if given
#   ylim=(min,max) : it allows us to determine the range of the y axis of our plot
#   style=[] : this allows us to define the style of each of the lines that are plotted
#              '-' will be a continuous line, '--' will be discrete line

# performance evaluation
MAE_ma=df_ma['Error'].abs().mean()
print("MAE of Moving Average Model:",round(MAE_ma,2))

RMSE_ma= np.sqrt((df_ma['Error']**2).mean())
print("RMSE of Moving Average Model:",round(RMSE_ma,2))

MAEper_ma=df_ma['Error'].abs().sum()/df_ma['Demand'][:len(d)].sum()
print("MAE% of Moving Average Model:",round(MAEper_ma,2))

#an example
df_ma[['Demand','Forecast']].plot(figsize=(8,3), title='Moving Average', ylim = (0,30), style=['-','-*'])

#by default, .plot() will use the DataFrame index as the x axis. Therefore, to change it, just name the DataFrame index.
df_ma.index.name = 'Period'

#2. Simple Exponential method

def simple_exp_smooth(d,extra_periods=1,alpha=0.4):
    #INPUTS
    #d: a time series that contains the historical demand
    #extra_periods: the number of periods we want to forecast in the future
    #n: the number of periods we will average
    #Transform the input into a numpy array
    d = np.array(d)
    #Historical period length
    cols = len(d)
    #append np.nan into the demand array to cover future periods
    d = np.append(d,[np.nan]*extra_periods)
    #define the forecast array
    f = np.full(cols + extra_periods, np.nan)
    #initialization of first forecast
    f[1]=d[0]

    #Create all the t+1 forecasts until end of historical period
    for t in range(2,cols+1):
        f[t]=alpha*d[t-1]+(1-alpha)*f[t-1]
  
    #Forecast for all extra periods
    f[cols+1:]=f[t]
    
    df= pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Error":d-f})
    
    return df

df_exp=simple_exp_smooth(d,extra_periods=4)

# performance evaluation
MAE_exp=df_exp['Error'].abs().mean()
print("MAE of Simple Exponential Model:",round(MAE_exp,2))

RMSE_exp=np.sqrt((df_exp['Error']**2).mean())
print("RMSE of Simple Exponential Model:",round(RMSE_exp,2))

MAEper_exp=(df_exp['Error'].abs().sum()) / df_exp['Demand'][:len(d)].sum()
print("MAE% of Simple Exponential Model:",round(MAEper_exp,2))

df_exp.index_name='Period'
df_exp[['Demand','Forecast']].plot(figsize=(8,3),title='Simple Smoothing',ylim=(0,30),style=['-','-*'])


# determine best alpha value regarding MAE    
MAE_dict={}
score_mae=[] 
alpha=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for i in alpha:
    df_exp=simple_exp_smooth(d,extra_periods=1,alpha=i)
    MAE_exp=df_exp['Error'].abs().mean()
    score_mae.append(MAE_exp)
    MAE_dict[alpha.index(i)]=round(score_mae[alpha.index(i)],3)
MAE_dict   

#3. Double Exponential method

def double_exp_smooth(d,extra_periods=1,alpha=0.4,beta=0.4):
    d=np.array(d) #transform the input into a numpy array
    cols=len(d) #historical period length
    d=np.append(d,[np.nan]*extra_periods) #append np.nan into the demand array to cover future periods
    
    #creation of the level, trend and forecast arrays
    f,a,b=np.full((3,cols+extra_periods),np.nan)
    
    #level & trend initialization
    a[0]=d[0]
    b[0]=d[1]-d[0]
    
    #create all the t+1 forecasts
    for t in range(1,cols):
        f[t]=a[t-1]+b[t-1]
        a[t]=alpha*d[t]+(1-alpha)*(a[t-1]+b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*b[t-1]

   #forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t]=a[t-1]+b[t-1]
        a[t]=f[t]
        b[t]=f[t-1]
    
    df=pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Level":a,"Trend":b,"Error":d-f})
    
    return df 

df_double=double_exp_smooth(d,extra_periods=4)

MAE=df_double['Error'].abs().mean()
print('MAE of Double Exponential Model',round(MAE,2))
RMSE=np.sqrt((df_double['Error']**2).mean())
print("RMSE of Double Exponential Model",round(RMSE,2))
MAEper_double=(df_double['Error'].abs().sum()) / df_double['Demand'][:len(d)].sum()
print("MAE% of Double Exponential Model:",round(MAEper_double,2))

df_double.index.name='Periods'
df_double[['Demand','Forecast']].plot(figsize=(8,3),title='Double Smoothing',ylim=(0,30),style=['-','-*'])

# Model Optimization - Optimize parameters 
def exp_smooth_opti(d,extra_periods=6):
    param=[] #contain all the different parameter sets
    KPI=[]  #contain the results of each model
    outputs=[] #contain all the dataframes returned by the different models
    
    for alpha in [0.05,0.1,0.2,0.3,0.4,0.5,0.6]:
        
           df=simple_exp_smooth(d,extra_periods=extra_periods,alpha=alpha)
           param.append("Simple Smoothing,alpha is:"+str(alpha))
           outputs.append(df)
           MAE=df['Error'].abs().mean()
           KPI.append(MAE)
        
           for beta in [0.05,0.1,0.2,0.3,0.4]:
                df=double_exp_smooth(d,extra_periods=extra_periods,alpha=alpha,beta=beta)
                param.append('Double Smoothing,alpha is:'+str(alpha)+", beta:"+str(beta))
                outputs.append(df)
                MAE=df['Error'].abs().mean()
                KPI.append(MAE)
    
    mini=np.argmin(KPI)
    print('Best solution found for',param[mini],'and MAE of',round(KPI[mini],2))
    return outputs[mini]

df=exp_smooth_opti(d)
df[['Demand','Forecast']].plot(figsize=(8,3),title='Best Model',ylim=(0,30),style=["-","-*"])

#Double Smmothing with Damped Trend

def double_exp_smooth_damped(d,extra_periods,alpha=0.4,beta=0.4,phi=0.9):
    d=np.array(d) #transform the input into a numpy array
    cols=len(d)
    d=np.append(d,[np.nan]*extra_periods) # append np.nan into the demand array to cover future periods
    
    #creation of the level, trend and forecast arrays
    f,a,b=np.full((3,cols+extra_periods),np.nan)
    
    #level & trend initialization
    a[0]=d[0]
    b[0]=d[1]-d[0]
    
    #create all the t+1 forecasts
    for t in range(1,cols):
        f[t]=a[t-1]+phi*b[t-1]
        a[t]=alpha*d[t]+(1-alpha)*(a[t-1]+phi*b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*phi*b[t-1]
    
    
    #forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t]=a[t-1]+phi*b[t-1]
        a[t]=f[t]
        b[t]=phi*b[t-1]
        
    df=pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Level':a,'Trend':b,'Error':d-f})
    
    return df

df_damped=double_exp_smooth_damped(d,extra_periods=4)

MAE=df_damped['Error'].abs().mean()
print('MAE of Double Exponential with Damped Trend',round(MAE,2))
RMSE=np.sqrt((df_damped['Error']**2).mean())
print("RMSE of Double Exponential with Damped Trend",round(RMSE,2))
MAEper_damped=(df_damped['Error'].abs().sum()) / df_damped['Demand'][:len(d)].sum()
print("MAE% of Double Exponential with Damped Trend:",round(MAEper_damped,2))

df_damped[['Demand','Forecast']].plot(figsize=(8,3),title='Double Exponential with Damped Trend',ylim=(0,30),style=["-","-*"])

#4.Multiplicative Triple Exponential Smoothing
def seasonal_factors_mul(s,d,slen,cols):
    for i in range(slen):
        idx=[x for x in range(cols) if x%slen==i] #compute indices that correspond to this season
        s[i]=np.mean(d[idx]) #compute season average
    s /=np.mean(s[:slen])
    return s

def triple_exp_smooth_mul(d,slen=12,extra_periods=1,alpha=0.4,beta=0.4,phi=0.9,gamma=0.3):
    d=np.array(d) #transform the input into a numpy array
    cols=len(d) #historical period length
    d=np.append(d,[np.nan]*extra_periods) #append np.nan into the demand array to cover future periods
    
    #components initialization
    f,a,b,s=np.full((4,cols+extra_periods),np.nan)
    s=seasonal_factors_mul(s,d,slen,cols)
    
    #level & trend initialization
    a[0]=d[0]/s[0]
    b[0]=d[1]/s[1]-d[0]/s[0]
    
    #create the forecast first season
    for t in range(1,slen):
        f[t]=(a[t-1]+phi*b[t-1])*s[t]
        a[t]=alpha*d[t]/s[t]+(1-alpha)*(a[t-1]+phi*b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*phi*b[t-1]
    
    #create all the t+1 forecasts
    for  t in range(slen,cols):
        f[t]=(a[t-1]+phi*b[t-1])*s[t-slen]
        a[t]=alpha*d[t]/s[t-slen]+(1-alpha)*(a[t-1]+phi*b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*phi*b[t-1]
        s[t]=gamma*d[t]/a[t]+(1-gamma)*s[t-slen]
    
    #forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t]=(a[t-1]+phi*b[t-1])*s[t-slen]      
        a[t]=f[t]/s[t-slen]
        b[t]=phi*b[t-1]
        s[t]=s[t-slen]
    
    df=pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Level':a,'Trend':b,'Season':s,'Error':d-f})
    return df

z=[14,10,6,2,18,8,4,1,16,9,5,3,18,11,4,2,17,9,5,1]
zf=triple_exp_smooth_mul(z)

MAE_triple_mul=zf['Error'].abs().mean()
print('MAE of Triple Exponential Model ',round(MAE_triple_mul,2))
RMSE_triple_mul=np.sqrt((zf['Error']**2).mean())
print("RMSE of Triple Exponential Model",round(RMSE_triple_mul,2))
MAEper_triple_mul=(zf['Error'].abs().sum()) / zf['Demand'][:len(d)].sum()
print("MAE% of Triple Exponential Model:",round(MAEper_triple_mul,2))

zf.plot(subplots=True,figsize=(16,7),title='Components of Multiplicative Holt Winters')
zf[['Level','Trend','Season']].plot(secondary_y=['Season'],figsize=(16,6),title='Multiplicative Triple Exponential Smoothing')

#4.Additive Triple Exponential Smoothing
def seasonal_factors_add(s,d,slen,cols):
    for i in range(slen):
        idx=[x for x in range(cols) if x%slen==i] #compute indices that correspond to this season
        s[i]=np.mean(d[idx]) #compute season average
    s-=np.mean(s[:slen]) #scale season factors (sum of factors=0)
    return s

def triple_exp_smooth_add(d,slen=10,extra_periods=1,alpha=0.4,beta=0.4,phi=0.9,gamma=0.3):
    d=np.array(d) #transform the input into a numpy array
    cols=len(d)  #historical period length
    d=np.append(d,[np.nan]*extra_periods) #append np.nan into the demand array to cover future periods
    
    #components initalization
    f,a,b,s=np.full((4,cols+extra_periods),np.nan)
    s=seasonal_factors_add(s, d, slen, cols)
    
    #Level&Trend initialization
    a[0]=d[0]-s[0]
    b[0]=(d[1]-s[1])-(d[0]-s[0])
    
    #create a forecast for the first season
    for t in range(1,slen):
        f[t]=a[t-1]+phi*b[t-1]+s[t]
        a[t]=alpha*(d[t]-s[t])+(1-alpha)*(a[t-1]+phi*b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*phi*b[t-1]
    
    #create all the t+1 forecasts
    for t in range (slen,cols):
        f[t]=a[t-1]+phi*b[t-1]+s[t-slen]
        a[t]=alpha*(d[t]-s[t-slen])+(1-alpha)*(a[t-1]+phi*b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*phi*b[t-1]
        s[t]=gamma*(d[t]-a[t])+(1-gamma)*s[t-slen]
 
    #forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t]=a[t-1]+phi*b[t-1]+s[t-slen]
        a[t]=f[t]-s[t-slen]
        b[t]=phi*b[t-1]
        s[t]=s[t-slen]
    
    df=pd.DataFrame.from_dict({'Demand':d,'Forecast':f,'Level':a,'Trend':b,'Season':s,'Error':d-f})
    return df
    
df_add=triple_exp_smooth_add(z)

MAE_triple_add=df_add['Error'].abs().mean()
print('MAE of Triple Exponential Model ',round(MAE_triple_add,2))
RMSE_triple_add=np.sqrt((df_add['Error']**2).mean())
print("RMSE of Triple Exponential Model",round(RMSE_triple_add,2))
MAEper_triple_add=(df_add['Error'].abs().sum()) / df_add['Demand'][:len(d)].sum()
print("MAE% of Triple Exponential Model:",round(MAEper_triple_add,2))

df_add.plot(subplots=True,figsize=(16,7),title='Components of Additive Holt Winters')
df_add[['Level','Trend','Season']].plot(secondary_y=['Season'],figsize=(16,6),title='Additive Triple Exponential Smoothing')




