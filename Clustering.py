# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import calendar

# Training and test set creation
def import_data():
    data=pd.read_csv('~/Documents/Demand Forecasting/norway_car_sales.csv')
    data['Period']=data['Year'].astype(str) + "-" +data['Month'].astype(str)
    data['Period']=pd.to_datetime(data['Period']).dt.strftime("%Y-%m")
    df=pd.pivot_table(data=data,values='Quantity',index='Make',columns='Period',aggfunc='sum',fill_value=0)
    return df

df=import_data()

# Seasonal Factor creation
def seasonal_factors(df):
    s=pd.DataFrame(index=df.index)
    for month in range(12):
        col=[x for x in range(df.shape[1]) if x%12==month] # column indices that match this month
        s[month+1]=np.mean(df.iloc[:,col],axis=1) # compute season average for this month
    s=s.divide(s.mean(axis=1),axis=0)
    return s

def scaler(s):
    mean=s.mean(axis=1)
    maxi=s.max(axis=1)
    mini=s.min(axis=1)
    s=s.subtract(mean,axis=0)
    s=s.divide(maxi-mini,axis=0).fillna(0)
    return s

s=seasonal_factors(df)
s_normalize=scaler(s)

kmeans=KMeans(n_clusters=6,random_state=0).fit(s_normalize)

df['Group']=kmeans.predict(s_normalize) # add the results back into df

#EXPERIMENTATION: 
# define the dataframe that will contain our results
results=pd.DataFrame(columns=['Inertia','Number of Clusters'])

for n in range(1,10):
    kmeans=KMeans(n_clusters=n,random_state=0).fit(s_normalize)
    results=results.append({'Inertia':kmeans.inertia_,'Number of Clusters':n},ignore_index=True)

#plot the results    
results.set_index('Number of Clusters').plot()

# VISUALIZATION:
centers=kmeans.cluster_centers_
centers=pd.DataFrame(centers,columns=range(1,13)).transpose()

# Cleaning & formatting
month_names=[calendar.month_abbr[month_idx] for month_idx in range(1,13)]
columns_names=['Cluster' + str(x) for x in range(centers.shape[1])]
centers.index=month_names
centers.columns=columns_names

sns.set(font_scale=2)
plt.figure(figsize=(16, 16))
sns.heatmap(centers,annot=True,center=0,cmap='RdBu_r')

print(df['Group'].value_counts().sort_index())
