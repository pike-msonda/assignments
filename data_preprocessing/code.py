import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def equal_width(data, info,interval):
    res = []
    for column in info:
        column_max = info[column]['max']
        column_min = info[column]['min']
        width = (column_max - column_min )/ interval
        bins = make_bins(column_min, interval, width)
        res.append(pd.cut(data[column], bins).value_counts())
    return res

def make_bins(min, interval, width):
    bins = []
    bins.append(min)
    count = 1
    while (count < interval):
            inter =  min + width
            bins.append(inter)
            min =  inter
            count = count  + 1
    return bins

# Read from sonar.dat file
sonar_data =  pd.read_table('sonar.dat',sep=',',usecols=[0,1,2,3,60])
# 5 Row Summary:
# print (sonar_data.head())
# # Mean
# print (sonar_data.mean())
# # Mode
# print (sonar_data.mode())
# #  Standard Deviation
# print (sonar_data.std())

# # Variance
# print (sonar_data.var())



# print (sonar_data.quantile([0.25,0.5,0.75]))
# # Min, Max Normalisation
# # Remove Type column. 
df = sonar_data.drop(['Type'], axis= 1)
normalised_data = (df - df.min())/(df.max() -  df.min())
print "Data after Min Max Normalisation"
print (normalised_data.head())
# Z-score Normalisation
z_normalised_data = (df - df.mean())/df.std()
print "Data after Z-score Normalisation"
print (z_normalised_data.head())

# Equal Width
info = sonar_data.describe()
print (info)
bins = equal_width(df,info,10)
# for col in df:
#     hist = sns.distplot(df['Band1'], bins=10)
#     plt.figure()
# Plot Boxplot for all attributes
# ax = sns.boxplot(data=normalised_data)
# ax = sns.swarmplot(data=normalised_data)
plt.show()       