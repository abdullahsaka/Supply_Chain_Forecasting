# Importing packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import datetime

# Loading datasets
org_train_dataset = pd.read_csv('~/Documents/Datasets/walmart-store-forecasting/train.csv',sep=',', header=0)
min(org_train_dataset.Date) # 2010-02-05
max(org_train_dataset.Date) # 2012-10-26
len(org_train_dataset[org_train_dataset['Weekly_Sales'] < 0]) # 1285 weekly sales below than 0
len(org_train_dataset[org_train_dataset['Weekly_Sales'] == 0]) # 73 weekly sales equal to 0
test_dataset = pd.read_csv('~/Documents/Datasets/walmart-store-forecasting/test.csv',sep=',', header=0)

# Check forecasting lag (what is starting forecasting week for each store & dept?)
train_time=org_train_dataset.groupby(['Store','Dept']).max()['Date']
train_time=pd.DataFrame(train_time).reset_index()
train_time=train_time.rename(columns={'Date':'FinishActual'})
test_time=test_dataset.groupby(['Store','Dept']).min()['Date']
test_time=pd.DataFrame(test_time).reset_index()
test_time=test_time.rename(columns={'Date':'StartForecast'})
all_time=pd.merge(left=train_time, right=test_time, left_on=['Store','Dept'], right_on=['Store','Dept'],how='outer')
all_time.StartForecast=pd.to_datetime(all_time.StartForecast).dt.date
all_time.FinishActual=pd.to_datetime(all_time.FinishActual).dt.date
all_time['Difference']=all_time.StartForecast-all_time.FinishActual
fore_all_SKU=all_time[all_time.Difference=='7 days']

##----------------------------------------------------------------------------------------------------
# Dept & Store Analysis in Walmart Stores for year 2011
# 92 dry grocery, 38 apparel, 72 electronics, 95 beverage and snacks, 90 dairy products
data_2011 = org_train_dataset[(org_train_dataset.Date>='2011-01-01') & (org_train_dataset.Date<='2011-12-31')]
data_2011 = data_2011[data_2011['Weekly_Sales'] >= 0]
upd_data_2011= data_2011[data_2011['Weekly_Sales'] >= 0]

store_dept_avg=upd_data_2011.groupby(['Store','Dept']).mean()['Weekly_Sales'].reset_index()
store_dept_avg=pd.DataFrame(store_dept_avg)
store_dept_avg = store_dept_avg.rename(columns={'Weekly_Sales':'Average'})

# convert weekly sales(negative and zero) to average of store & dept
update_train_data = pd.merge(left=org_train_dataset, right=store_dept_avg, left_on=['Store','Dept'], right_on=['Store','Dept'],how='left')

def sales(Weekly_Sales,Average):
    if Weekly_Sales<=0:
        return Average
    else:
        return Weekly_Sales
    
update_train_data['Sales']=update_train_data.apply(lambda x: sales(x['Weekly_Sales'],x['Average']), axis=1)
update_train_data['Sales'].isnull().sum()
data_null=update_train_data[update_train_data.isnull().any(axis=1)]
train_dataset=update_train_data.copy()

# Find the releveant stores & departments in Walmart Demand Planning System
dept_sum=data_2011.groupby('Dept').sum()['Weekly_Sales'].reset_index()
dept_sum=pd.DataFrame(dept_sum)
dept_total=dept_sum['Weekly_Sales'].sum()
dept_sum['Ratio']=dept_sum['Weekly_Sales']/dept_total
dept_sum=dept_sum.sort_values(ascending=False,by=['Weekly_Sales']).reset_index(drop=True)
dept_sum['CumRatio']=dept_sum['Ratio'].cumsum()
imp_dept=dept_sum[dept_sum.CumRatio<0.95]
forecast_dept=imp_dept[['Dept']]
irr_dept=dept_sum[dept_sum.CumRatio>0.95]
irrelevant_dept=irr_dept[['Dept']]

sns.set(font_scale=3.25)
plt.figure(figsize=(50, 10))
plt.title('Department Sales Amount ($) across all stores',weight="bold")
plt.ticklabel_format(style='plain', axis='y')
spacing = 5
ax = sns.barplot(x="Dept", y="Weekly_Sales", data=dept_sum)
ax.set(xlabel='Department', ylabel='Total')
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels():
    if label not in visible:
        label.set_visible(False)

store_sum=data_2011.groupby('Store').sum()['Weekly_Sales'].reset_index()
store_sum=pd.DataFrame(store_sum)
store_total=store_sum['Weekly_Sales'].sum()
store_sum['Ratio']=store_sum['Weekly_Sales']/store_total
store_sum=store_sum.sort_values(ascending=False,by=['Weekly_Sales']).reset_index(drop=True)
store_sum['CumRatio']=store_sum['Ratio'].cumsum()
imp_store=store_sum[store_sum.CumRatio<0.95]
forecast_store=imp_store[['Store']]
irr_store=store_sum[store_sum.CumRatio>0.95]
irrelevant_store=irr_store[['Store']]

sns.set(font_scale=3)
plt.figure(figsize=(50, 10))
plt.title('Store Sales Amount($) across all stores',weight="bold")
plt.ticklabel_format(style='plain', axis='y')
spacing = 5
ax = sns.barplot(x="Store", y="Weekly_Sales", data=store_sum)
ax.set(xlabel='Store', ylabel='Total')
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels():
    if label not in visible:
        label.set_visible(False)
          
#-------------------------------------------------------------------------------------------------------
#Granularity Analysis(store=4 and dept=72)
sample_data=train_dataset[(train_dataset.Store==4) & (train_dataset.Dept==72)]
sns.set(font_scale=3)
plt.figure(figsize=(50, 10))
plt.title('Weekly Sales ($) of store 4 at department 72', weight='bold')
plt.xticks(rotation=45)
spacing = 10
ax=sns.lineplot(x="Date", y="Weekly_Sales", data=sample_data)
ax.set(xlabel='Date', ylabel='Weekly Sales')
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels():
    if label not in visible:
        label.set_visible(False)      
# -----------------------------------------------------------------------------------------------------
# Calculate train data size for each store & department combinations
period_length=[]
store_index=[]
dept_index=[]
for i in train_dataset.Store.unique():
    for j in train_dataset.Dept.unique():
            store_index += [i]
            dept_index += [j]
            perleng=len(train_dataset[(train_dataset.Store==i) & (train_dataset.Dept==j)])
            df=train_dataset[(train_dataset.Store==i) & (train_dataset.Dept==j)]['Weekly_Sales']
            period_length += [perleng]
            
pd.set_option('display.max_rows', None)
store_index = pd.DataFrame(store_index)
store_index=store_index.rename(columns={0:'Store'})
dept_index = pd.DataFrame(dept_index)
dept_index=dept_index.rename(columns={0:'Dept'})
period_length =pd.DataFrame(period_length)
train_data_size=pd.concat([store_index,dept_index,period_length],axis=1)
train_data_size=train_data_size.rename(columns={0:'Size'})
train_data_size=train_data_size[train_data_size.Size>0]

# Cluster combinations of store & dept  into 3 categories
def fore_class(Size):
    if Size<52:
        return 1
    elif 52<=Size<104 :
        return 2
    else:
        return 3
train_data_size['Model_class']=train_data_size.apply(lambda x: fore_class(x['Size']), axis=1)

# 1. Important stores and departments
upd_fore_data=pd.merge(left=train_data_size, right=imp_dept, left_on=['Dept'], right_on=['Dept'],how='inner')
upd_fore_data=pd.merge(left=upd_fore_data, right=imp_store, left_on=['Store'], right_on=['Store'],how='inner')
upd_fore_data=upd_fore_data[['Store','Dept','Size','Model_class']]

upd_fore_data=pd.merge(left=upd_fore_data, right=fore_all_SKU, left_on=['Store','Dept'], right_on=['Store','Dept'],how='inner')
upd_fore_data=upd_fore_data.drop(['Difference'],axis=1)

# 2. Irrelevant stores and departments
irv_fore_data=pd.merge(left=train_data_size, right=irrelevant_dept, left_on=['Dept'], right_on=['Dept'],how='inner')
irv_fore_data=pd.merge(left=irv_fore_data, right=irrelevant_store, left_on=['Store'], right_on=['Store'],how='inner')
irv_fore_data=irv_fore_data[['Store','Dept','Size','Model_class']]
irv_fore_data = irv_fore_data[irv_fore_data.Size>=13]

irv_fore_data=pd.merge(left=irv_fore_data, right=fore_all_SKU, left_on=['Store','Dept'], right_on=['Store','Dept'],how='inner')
irv_fore_data=irv_fore_data.drop(['Difference'],axis=1)
irv_fore_data['Code'] = irv_fore_data['Store'].map(str) + irv_fore_data['Dept'].map(str)

# Calculate test data size for each store & department combinations
period_length=[]
store_index=[]
dept_index=[]
for i in test_dataset.Store.unique():
    for j in test_dataset.Dept.unique():
            store_index += [i]
            dept_index += [j]
            perleng=len(test_dataset[(test_dataset.Store==i) & (test_dataset.Dept==j)])
            df=test_dataset[(test_dataset.Store==i) & (test_dataset.Dept==j)]['Date']
            period_length += [perleng]
                     
pd.set_option('display.max_rows', None)
store_index = pd.DataFrame(store_index)
store_index=store_index.rename(columns={0:'Store'})
dept_index = pd.DataFrame(dept_index)
dept_index=dept_index.rename(columns={0:'Dept'})
period_length =pd.DataFrame(period_length)
test_data_size=pd.concat([store_index,dept_index,period_length],axis=1)
test_data_size=test_data_size.rename(columns={0:'Size'})
test_data_size=test_data_size[test_data_size.Size>0]
# ------------------------------------------------------------------------------------------------------
# Create a forecast model regarding length of time series data
simple_exp_strdept = upd_fore_data[upd_fore_data['Model_class']==1] # In total, 40 combinations
simple_exp_strdept = simple_exp_strdept[simple_exp_strdept.Size>=13] # Removed 22 combinations
simple_exp_strdept['Code'] = simple_exp_strdept['Store'].map(str) + simple_exp_strdept['Dept'].map(str)


double_exp_strdept = upd_fore_data[upd_fore_data['Model_class']==2] 
double_exp_strdept['Code'] = double_exp_strdept['Store'].map(str) + double_exp_strdept['Dept'].map(str)


triple_exp_strdept = upd_fore_data[upd_fore_data['Model_class']==3] 
triple_exp_strdept['Code'] = triple_exp_strdept['Store'].map(str) + triple_exp_strdept['Dept'].map(str)
# ------------------------------------------------------------------------------------------------------
# 1.Moving Average Model
def moving_average(z,extra_periods,n):
    m=z.Date 
    d=z.Sales
    d = np.array(d)
    cols = len(d)
    d = np.append(d,[np.nan]*extra_periods)
    f = np.full(cols + extra_periods, np.nan)
    m=np.append(m,[np.nan]*extra_periods)
    
    y=np.full((cols+extra_periods),np.nan)
    y[0]=0
    
    for t in range(0,n):
        y[t]=0

    for t in range(n,cols+1):
        f[t] = np.mean(d[t-n:t])
        y[t-1]=0
    
    f[cols+1:] = f[t]
    y[cols+1:]=1
    
    df = pd.DataFrame.from_dict({"Date":m,'Demand':d,'Forecast':f,"FLag":y,'Error':d-f})
     
    return df

# Run Moving Average Model
moving_model={}
MAEper_ma={}
for i in irv_fore_data.Store.unique():
    for j in irv_fore_data.Dept.unique():
        SKU_code=str(i) + str(j)
        df=train_dataset[(train_dataset.Store==i) & (train_dataset.Dept==j)]
        df=df.sort_values(by='Date',ascending=True)
        if SKU_code in irv_fore_data['Code'].unique():
            size=len(df)-1
            df_ma=moving_average(df,39,5)
            for h in range(0,40):
                df_ma.Date = pd.to_datetime(df_ma.Date).dt.date
                df_ma['Date'][size+h] = df_ma['Date'][size] + datetime.timedelta(weeks=+h)
            MAEper=(df_ma['Error'][len(df)-13:len(df)].abs().sum()/df_ma['Demand'][len(df)-13:len(df)].sum())*100
            MAEper_ma[i,j]=round(MAEper,5)
            moving_model[i,j]=df_ma
  
# Evaluate performance of the Moving Average Model      
store=[(k[0]) for k,v in MAEper_ma.items()]
store=pd.DataFrame(store,columns=['Store'])
dept=[(k[1]) for k,v in MAEper_ma.items()]
dept=pd.DataFrame(dept,columns=['Dept'])
MAE_per=[v for k,v in MAEper_ma.items()]
MAE_per=pd.DataFrame(MAE_per,columns=['MovingAvg_Result'])
moving_avg_performance=pd.concat([store,dept,MAE_per],axis=1)
moving_avg_performance.dropna(subset=['MovingAvg_Result'], how='all', inplace=True)
moving_avg_performance.head()   

final_moving=pd.merge(left=irv_fore_data, right=moving_avg_performance, left_on=['Store','Dept'], right_on=['Store','Dept'],how='left')
final_moving=final_moving.rename(columns={'MovingAvg_Result':'Forecast_Error'})
final_moving['Forecast_Accuracy']=100-final_moving['Forecast_Error']
final_moving['Method']='MovingAvg'
final_moving.head()

# Build the final table shows components of the forecasting model
nrow=len(moving_model.keys())
moving_avg_forecast=pd.DataFrame()
for i in range(nrow):
    df_inter=moving_model[list(moving_model.keys())[i]]
    df_inter=df_inter.assign(Store=list(moving_model.keys())[i][0])
    df_inter=df_inter.assign(Dept=list(moving_model.keys())[i][1])
    moving_avg_forecast=moving_avg_forecast.append(df_inter)
moving_avg_forecast=moving_avg_forecast.reset_index(drop=True)

# Granularity Analysis: Actual vs Prediction (sample: store 30 - dept 83)
plot_moving_data=moving_avg_forecast[(moving_avg_forecast.Store==30) & (moving_avg_forecast.Dept==83)].reset_index(drop=True)
plot_moving_data=plot_moving_data[:142]
sns.set(font_scale=3)
plt.figure(figsize=(45, 10)) # change figure size
plt.title('Actual vs Forecasting for Dept 83 on Store 30', weight='bold')
spacing = 16
ax=sns.lineplot(x="Date", y="Forecast", data=plot_moving_data,color='blue')
ax=sns.lineplot(x="Date", y="Demand", data=plot_moving_data,color='red')
ax.set(xlabel='Date', ylabel='Weekly Sales')
ax.set(xticks=plot_moving_data.Date.values)
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels(): # arrange x axis
    if label not in visible:
        label.set_visible(False)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=32)
ax.legend(('Predict','Actual'),loc='upper left') # add a legend to plot
plt.savefig('Store-30-Dept-83.png') #save figure
plt.show()
# ------------------------------------------------------------------------------------------------------        
# 2.Simple Exponential Model     
def simple_exp_smooth(z,extra_periods,alpha):
    #INPUTS
    #d: a time series that contains the historical demand
    #extra_periods: the number of periods we want to forecast in the future
    #n: the number of periods we will average
    #Transform the input into a numpy array
    m=z.Date
    d=z.Sales
    d = np.array(d)
    #Historical period length
    cols = len(d)
    #append np.nan into the demand array to cover future periods
    d = np.append(d,[np.nan]*extra_periods)
    m=np.append(m,[np.nan]*extra_periods)
    #define the forecast array
    f = np.full(cols + extra_periods, np.nan)
    #initialization of first forecast
    f[1]=d[0]

    z=np.full((cols+extra_periods),np.nan)
    z[0]=0

    #Create all the t+1 forecasts until end of historical period
    for t in range(2,cols+1):
        f[t]=alpha*d[t-1]+(1-alpha)*f[t-1]
        z[t-1]=0
        if t==cols:
           z[t]=1
  
    #Forecast for all extra periods
    f[cols+1:]=f[t]
    z[cols+1:]=1
    
    df= pd.DataFrame.from_dict({"Date":m,"Demand":d,"Forecast":f,"Flag":z,"Error":d-f})
    
    return df

# Run Simple Exponential Model
MAEper_simple={}
simple_model={}
for i in simple_exp_strdept.Store.unique():
    for j in simple_exp_strdept.Dept.unique():        
        SKU_code=str(i) + str(j)
        df=train_dataset[(train_dataset.Store==i) & (train_dataset.Dept==j)]
        size=len(df)-1
        df=df.sort_values(by='Date',ascending=True)
        if SKU_code in simple_exp_strdept['Code'].unique():
           df_exp = simple_exp_smooth(df,39,0.4)
           for h in range(0,40):
                df_exp.Date = pd.to_datetime(df_exp.Date).dt.date
                df_exp['Date'][size+h] = df_exp['Date'][size] + datetime.timedelta(weeks=+h)
           MAEper = (df_exp['Error'][len(df)-13:len(df)].abs().sum()/df_exp['Demand'][len(df)-13:len(df)].sum())*100
           MAEper_simple[i,j] = round(MAEper,3)
           simple_model[i,j]=df_exp
        else:   
            MAEper_simple[i,j]=0
                
# Evaluate performance of the Simple Exponential Model   
store=[(k[0]) for k,v in MAEper_simple.items()]
store=pd.DataFrame(store,columns=['Store'])
dept=[(k[1]) for k,v in MAEper_simple.items()]
dept=pd.DataFrame(dept,columns=['Dept'])
MAE_per_s=[v for k,v in MAEper_simple.items()]
MAE_per_s=pd.DataFrame(MAE_per_s,columns=['SimpleExp_Result'])
simple_performance=pd.concat([store,dept,MAE_per_s],axis=1)
simple_performance.dropna(subset=['SimpleExp_Result'], how='all', inplace=True)
simple_performance.head()

# remove non-existent store & dept combinations
final_simple=pd.merge(left=simple_exp_strdept, right=simple_performance, left_on=['Store','Dept'], right_on=['Store','Dept'],how='left')
final_simple=final_simple.rename(columns={'SimpleExp_Result':'Forecast_Error'})
final_simple['Forecast_Accuracy']=100-final_simple['Forecast_Error']
final_simple.head()

# Build the final table shows components of the forecasting model
nrow=len(simple_model.keys())
simple_forecast=pd.DataFrame()
for i in range(nrow):
    df_inter=simple_model[list(simple_model.keys())[i]]
    df_inter=df_inter.assign(Store=list(simple_model.keys())[i][0])
    df_inter=df_inter.assign(Dept=list(simple_model.keys())[i][1])
    simple_forecast=simple_forecast.append(df_inter)
simple_forecast=simple_forecast.reset_index(drop=True)

# Granularity Analysis: Actual vs Prediction (sample: store 17 - dept 96)
plot_simple_data=simple_forecast[(simple_forecast.Store==17) & (simple_forecast.Dept==96)].reset_index(drop=True)
plot_simple_data=plot_simple_data[:47]
sns.set(font_scale=3)
plt.figure(figsize=(45, 10)) # change figure size
plt.title('Actual vs Forecasting for Dept 96 on Store 17', weight='bold')
spacing = 8
ax=sns.lineplot(x="Date", y="Forecast", data=plot_simple_data,color='blue')
ax=sns.lineplot(x="Date", y="Demand", data=plot_simple_data,color='red')
ax.set(xlabel='Date', ylabel='Weekly Sales')
ax.set(xticks=plot_simple_data.Date.values)
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels(): # arrange x axis
    if label not in visible:
        label.set_visible(False)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=32)
ax.legend(('Predict','Actual'),loc='upper left') # add a legend to plot
plt.savefig('Store-17-Dept-96.png') #save figure
plt.show()
# ------------------------------------------------------------------------------------------------------
# 3.Double Exponential Model   
def double_exp_smooth(z,extra_periods,alpha=0.4,beta=0.4):
    m=z.Date
    d=z.Weekly_Sales
    d=np.array(d) #transform the input into a numpy array
    cols=len(d) #historical period length
    m=np.append(m,[np.nan]*extra_periods)
    d=np.append(d,[np.nan]*extra_periods) #append np.nan into the demand array to cover future periods
    
    #creation of the level, trend and forecast arrays
    f,a,b,z=np.full((4,cols+extra_periods),np.nan)
    
    #level & trend initialization
    a[0]=d[0]
    b[0]=d[1]-d[0]
    z[0]=0
    
    #create all the t+1 forecasts
    for t in range(1,cols):
        f[t]=a[t-1]+b[t-1]
        a[t]=alpha*d[t]+(1-alpha)*(a[t-1]+b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*b[t-1]
        z[t]=0

   #forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t]=a[t-1]+b[t-1]
        a[t]=f[t]
        b[t]=f[t-1]
        z[t]=1
    
    df=pd.DataFrame.from_dict({"Date":m,"Demand":d,"Forecast":f,"FLag":z,"Level":a,"Trend":b,"Error":d-f})
    
    return df
  
# Run Double Exponential Model   
MAEper_double={}
double_model={}
for i in double_exp_strdept.Store.unique():
    for j in double_exp_strdept.Dept.unique():
        SKU_code=str(i) + str(j)
        df=train_dataset[(train_dataset.Store==i) & (train_dataset.Dept==j)]
        size=len(df)-1
        df=df.sort_values(by='Date',ascending=True)
        if SKU_code in double_exp_strdept['Code'].unique():
            df_double=double_exp_smooth(df,39,0.4,0.4)
            for h in range(0,40):
                df_double.Date = pd.to_datetime(df_double.Date).dt.date
                df_double['Date'][size+h] = df_double['Date'][size] + datetime.timedelta(weeks=+h)
            MAEper=(df_double['Error'][len(df)-13:len(df)].abs().sum()/df_double['Demand'][len(df)-13:len(df)].sum())*100
            MAEper_double[i,j]=round(MAEper,5)
            double_model[i,j]=df_double
        else:
            MAEper_double[i,j]=0
                      
# Evaluate performance of the Simple Exponential Model
store=[(k[0]) for k,v in MAEper_double.items()]
store=pd.DataFrame(store,columns=['Store'])
dept=[(k[1]) for k,v in MAEper_double.items()]
dept=pd.DataFrame(dept,columns=['Dept'])
MAE_per_d=[v for k,v in MAEper_double.items()]
MAE_per_d=pd.DataFrame(MAE_per_d,columns=['DoubleExp_Result'])
double_performance=pd.concat([store,dept,MAE_per_d],axis=1)
double_performance.dropna(subset=['DoubleExp_Result'], how='all', inplace=True)
double_performance.head()

# remove non-existent store & dept combinations
final_double=pd.merge(left=double_exp_strdept, right=double_performance, left_on=['Store','Dept'], right_on=['Store','Dept'],how='left')
final_double=final_double.rename(columns={'DoubleExp_Result':'Forecast_Error'})
final_double['Forecast_Accuracy']=100-final_double['Forecast_Error']
final_double.head()

# Build the final table shows components of the forecasting model
nrow=len(double_model.keys())
double_forecast=pd.DataFrame()
for i in range(nrow):
    df_inter=double_model[list(double_model.keys())[i]]
    df_inter=df_inter.assign(Store=list(double_model.keys())[i][0])
    df_inter=df_inter.assign(Dept=list(double_model.keys())[i][1])
    double_forecast=double_forecast.append(df_inter)
double_forecast=double_forecast.reset_index(drop=True)

# Granularity Analysis: Actual vs Prediction (sample: store 16 - dept 93)
plot_double_data=double_forecast[(double_forecast.Store==16) & (double_forecast.Dept==93)].reset_index(drop=True)
sns.set(font_scale=3)
plt.figure(figsize=(45, 10)) # change figure size
plt.title('Actual vs Forecasting for Dept 93 on Store 16', weight='bold')
spacing = 6
ax=sns.lineplot(x="Date", y="Forecast", data=plot_double_data,color='blue')
ax=sns.lineplot(x="Date", y="Demand", data=plot_double_data,color='red')
ax.set(xlabel='Date', ylabel='Weekly Sales')
ax.set(xticks=plot_double_data.Date.values)
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels(): # arrange x axis
    if label not in visible:
        label.set_visible(False)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=32)
ax.legend(('Predict','Actual'),loc='upper left') # add a legend to plot
plt.savefig('Store-16-Dept-93.png') #save figure
plt.show()

# Granularity Analysis: Actual vs Prediction (sample: store 16 - dept 93)
plot_double_data=double_forecast[(double_forecast.Store==16) & (double_forecast.Dept==93)].reset_index(drop=True)
plot_double_data=plot_double_data[1:56]
sns.set(font_scale=3)
plt.figure(figsize=(45, 10)) # change figure size
plt.title('Actual vs Forecasting for Dept 93 on Store 16', weight='bold')
spacing = 6
ax=sns.lineplot(x="Date", y="Forecast", data=plot_double_data,color='blue')
ax=sns.lineplot(x="Date", y="Demand", data=plot_double_data,color='red')
ax.set(xlabel='Date', ylabel='Weekly Sales')
ax.set(xticks=plot_double_data.Date.values)
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels(): # arrange x axis
    if label not in visible:
        label.set_visible(False)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=32)
ax.legend(('Predict','Actual'),loc='upper left') # add a legend to plot
plt.savefig('Store-16-Dept-93.png') #save figure
plt.show()
# ------------------------------------------------------------------------------------------------------
# 4.Multiplicative Triple Exponential Smoothing
def seasonal_factors_mul(s,d,slen,cols):
    for i in range(slen):
        idx=[x for x in range(cols) if x%slen==i] #compute indices that correspond to this season
        s[i]=np.mean(d[idx]) #compute season average
    s /=np.mean(s[:slen])
    return s

def triple_exp_smooth_mul(z,slen=52,extra_periods=39,alpha=0.4,beta=0.4,phi=0.8,gamma=0.3):
    #m=pd.DataFrame(pd.to_datetime(z.Date).dt.date)
    m=z.Date
    d=z.Sales
    d=np.array(d) #transform the input into a numpy array
    cols=len(d) #historical period length
    d=np.append(d,[np.nan]*extra_periods) #append np.nan into the demand array to cover future periods
    m=np.append(m,[np.nan]*extra_periods)
    #components initialization
    f,a,b,s,z=np.full((5,cols+extra_periods),np.nan)
    s=seasonal_factors_mul(s,d,slen,cols)
    
    #level & trend initialization
    a[0]=d[0]/s[0]
    b[0]=d[1]/s[1]-d[0]/s[0]
    z[0]=0

    #create the forecast first season
    for t in range(1,slen):
        f[t]=(a[t-1]+phi*b[t-1])*s[t]
        a[t]=alpha*d[t]/s[t]+(1-alpha)*(a[t-1]+phi*b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*phi*b[t-1]
        z[t]=0
    
    #create all the t+1 forecasts
    for  t in range(slen,cols):
        f[t]=(a[t-1]+phi*b[t-1])*s[t-slen]
        a[t]=alpha*d[t]/s[t-slen]+(1-alpha)*(a[t-1]+phi*b[t-1])
        b[t]=beta*(a[t]-a[t-1])+(1-beta)*phi*b[t-1]
        s[t]=gamma*d[t]/a[t]+(1-gamma)*s[t-slen]
        z[t]=0

    #forecast for all extra periods
    for t in range(cols,cols+extra_periods):
        f[t]=(a[t-1]+phi*b[t-1])*s[t-slen]      
        a[t]=f[t]/s[t-slen]
        b[t]=phi*b[t-1]
        s[t]=s[t-slen]
        z[t]=1
        
    df=pd.DataFrame.from_dict({'Date':m,'Demand':d,'Forecast':f,'Flag':z,'Level':a,'Trend':b,'Season':s,'Error':d-f})
    return df

# Run Multiplicative Triple Exponential Model 
MAEper_triple={}
forecast_model={}
for i in triple_exp_strdept.Store.unique():
    for j in triple_exp_strdept.Dept.unique():
        SKU_code=str(i) + str(j)
        df=train_dataset[(train_dataset.Store==i) & (train_dataset.Dept==j)]
        size=len(df)-1
        df=df.sort_values(by='Date',ascending=True)
        if SKU_code in triple_exp_strdept['Code'].unique():
            df_triple=triple_exp_smooth_mul(df)
            for h in range(0,40):
                df_triple.Date = pd.to_datetime(df_triple.Date).dt.date
                df_triple['Date'][size+h] = df_triple['Date'][size] + datetime.timedelta(weeks=+h)
            MAEper=(df_triple['Error'][len(df)-13:len(df)].abs().sum()/df_triple['Demand'][len(df)-13:len(df)].sum())*100
            MAEper_triple[i,j]=round(MAEper,5)
            forecast_model[i,j]=df_triple
        else:   
            MAEper_triple[i,j]=0

# Evaluate performance of the Multiplicative Triple Exponential Model
store=[(k[0]) for k,v in MAEper_triple.items()]
store=pd.DataFrame(store,columns=['Store'])
dept=[(k[1]) for k,v in MAEper_triple.items()]
dept=pd.DataFrame(dept,columns=['Dept'])
MAE_per_trip=[v for k,v in MAEper_triple.items()]
MAE_per_trip=pd.DataFrame(MAE_per_trip,columns=['TripleExp_Result'])
triple_performance=pd.concat([store,dept,MAE_per_trip],axis=1)
triple_performance.dropna(subset=['TripleExp_Result'], how='all', inplace=True)
triple_performance.head()

# Build a table indicates forecasting error of each store & dept combination
final_triple=pd.merge(left=triple_exp_strdept, right=triple_performance, left_on=['Store','Dept'], right_on=['Store','Dept'],how='left')
final_triple=final_triple.rename(columns={'TripleExp_Result':'Forecast_Error'})
final_triple['Forecast_Accuracy']=100-final_triple['Forecast_Error']
final_triple.head()

# Build the final table shows components of the forecasting model
nrow=len(forecast_model.keys())
forecast=pd.DataFrame()
for i in range(nrow):
    df_inter=forecast_model[list(forecast_model.keys())[i]]
    df_inter=df_inter.assign(Store=list(forecast_model.keys())[i][0])
    df_inter=df_inter.assign(Dept=list(forecast_model.keys())[i][1])
    forecast=forecast.append(df_inter)
forecast=forecast.reset_index(drop=True)

# Plot the actual and predicted values(sample: store 32 - dept 90)
plot_data=forecast[(forecast.Store==32) & (forecast.Dept==90)].reset_index(drop=True)
sns.set(font_scale=3)
plt.figure(figsize=(45, 10))
plt.title('Actual vs Forecasting for Dept 90 on Store 32', weight='bold')
spacing = 16
ax=sns.lineplot(x="Date", y="Forecast", data=plot_data,color='blue')
ax=sns.lineplot(x="Date", y="Demand", data=plot_data,color='red')
ax.set(xlabel='Date', ylabel='Weekly Sales')
ax.set(xticks=plot_data.Date.values)
visible = ax.xaxis.get_ticklabels()[::spacing]
for label in ax.xaxis.get_ticklabels():
    if label not in visible:
        label.set_visible(False)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=32)
ax.legend(('Predict','Actual'),loc='upper left')
plt.savefig('Store-32-Dept-90.png')
plt.show()
#------------------------------------------------------------------------------------------------------------------------------           
# VALIDATION: Implement models in Excel and check whether they give same results or not
df=train_dataset[(train_dataset.Store==1) & (train_dataset.Dept==1)]
df=df.sort_values(by='Date',ascending=True) # must have ascending order by Date
df_modfy=df[['Date','Sales']] # take only data and historical sales
df_modfy.to_csv('Sample Validation.csv',index=False)
#------------------------------------------------------------------------------------------------------------------------------
# RESULT: Final Table
# combine all performance tables coming from different forecasting models
final_forecast_performance=pd.concat([final_moving,final_simple, final_double, final_triple], axis=0,ignore_index=True)
final_forecast_performance=final_forecast_performance.sort_values(ascending=True,by=['Store','Dept']).reset_index(drop=True)
final_forecast_performance=final_forecast_performance.drop(['Code','FinishActual','StartForecast','Model_class'],axis=1)
final_forecast_performance=final_forecast_performance[final_forecast_performance['Forecast_Accuracy']>0] # take observations whose accuracy is higher than 0
final_forecast_performance['Forecast_Accuracy'].describe()
final_forecast_performance['Forecast_Error'].describe()

# plot forecast accuracy of all store & dept combinations
final_forecast_performance['Forecast_Accuracy'].describe()
sns.set(font_scale=2)
sns.set_color_codes()
plt.figure(figsize=(12, 6))
plt.title('Distribution of Forecast Accuracy', weight='bold')
ax=sns.distplot(final_forecast_performance['Forecast_Accuracy'],color='navy',bins=50)
ax.set(xlabel='Accuracy')
plt.savefig('Accuracy.png') #save figure
plt.show()