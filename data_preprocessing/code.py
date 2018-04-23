import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

def equal_width(data, info,interval):
    for column in info:
        data[column] = pd.cut(data[column], interval,labels=create_label(interval))
    return data

def remove_classes(data):
    return data.drop(['Type'], axis = 1)

def create_label(bins):
    labels = []
    for x in range(0, bins):
        labels.append("bin"+str(x))
    return labels 
def probability(X):
   
    return X.value_counts(normalize=True, sort=False)

def cond_P(A,B):
    cond_events = A.groupby(B, sort=False)
    values = cond_events.value_counts(normalize=False).unstack(level=1).fillna(0)
    prob = []
    for col in values:
        total = sum(values[col])
        r = float(values[col][0])
        m = float(values[col][1])
        prob.append([np.divide(r,total), np.divide(m,total)])
    return prob

def entropys(A):
    total_ent = []
    for a in A:
        if 0 in a:
            ix = a.index(0)
            a[ix] = 1
        total_ent.append(entropy(a, base=2))
    return total_ent
def information_gain(data,types):
    gain = []
    data['Type'] = types
    size= data.index.size
    cases = data.groupby('Type').count().values
    n_rcases = float(cases.min())
    n_mcases = float(cases.max())
    total_prob = [n_mcases/size, n_rcases/size]
    total_ent = entropy(total_prob, base= 2)
    data = remove_classes(data)
    for col in data:
        counts =  data[col].value_counts().values
        counts = [float(x) for x in counts]
        probs = [x / size for x in counts]
        attribute_entropy = entropys(cond_P(data[col],types))
        info_gain = total_ent - sum([a * b for a,b in zip(attribute_entropy, probs)])
        gain.append({col: info_gain})
        
    return gain

      
if(__name__ == "__main__"):
    """
    Main program
    """
    # Read from sonar.dat file
    sonar_data =  pd.read_table('sonar.dat',sep=',',usecols=[0,1,2,3,60])
    df = remove_classes(sonar_data) # Temporarily remove the type column
    types = sonar_data['Type']
    n = 3
    #5 Row Summary:
    #print (sonar_data.head())
    
    # Mean, Mode, Standard Deviation, Variance
    #print (df.agg(['mean','std','var','min', 'max', 'count',np.percentile(25), ]))
    # print (sonar_data.quantile([0.25,0.5,0.75]))
    # # Min, Max Normalisation
    # # Remove Type column. 
    
    # normalised_data = (df - df.min())/(df.max() -  df.min())
    # print "Data after Min Max Normalisation"
    # print (normalised_data.head())
    # # Z-score Normalisation
    # z_normalised_data = (df - df.mean())/df.std()
    # print "Data after Z-score Normalisation"
    # print (z_normalised_data.head())

    # # Equal Width
    info = sonar_data.describe()
    #print (type(info))
    # bins = equal_width(df,info,10)
    # for col in df:
    #     hist = sns.distplot(df['Band1'], bins=10)
    #     plt.figure()
    # Plot Boxplot for all attributes
    # ax = sns.boxplot(data=normalised_data)
    # ax = sns.swarmplot(data=normalised_data)

    #information gain
    print(df.head())
    eqw= equal_width(df,info, n)
    info_gain = information_gain(eqw, types)
    print ("Information Gain using {} Equal -Width method".format(n))
    print (info_gain)     