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

def make_bins(min, interval, width):
    bins = []
    count = 1
    while (count < interval):
            inter =  min + width
            bins.append(pd.Interval(min, inter))
            min =  inter
            count = count  + 1
    return pd.IntervalIndex.from_intervals(bins)
def create_label(bins):
    labels = []
    for x in range(0, bins):
        labels.append("bin"+str(x))
    return labels 
def probability(X):
   
    return X.value_counts(normalize=True, sort=False)
def condition_P(A,B):
    cond_events = A.groupby(B, sort=False)
    return probability(cond_events)
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

def entropy_test(A):
    return  -sum(A * np.log2(A))
def info_gain(main_entropy, entropy):
    return main_entropy - entropy
def entropys(A):
    total_ent = []
    for a in A:
        if 0 in a:
            ix = a.index(0)
            a[ix] = 1
        total_ent.append(entropy(a, base=2))
    return total_ent
def information_gain(data,types):
    df['Type'] = types
    total= data.shape[0]
    cases = df.groupby('Type').count().values
    n_rcases = float(cases.min())
    n_mcases = float(cases.max())
    total_prob = [n_mcases/total, n_rcases/total]
    total_ent = entropy(total_prob, base= 2)
    print (total_ent)
    for col in data:
        counts =  data[col].value_counts().values
        counts = [float(x) for x in counts]
        probs = [x / total for x in counts]
        col_entropy = entropys(cond_P(data[col],types))
        print (probs)
        print (col_entropy)
        norm = 0
        for ent, p in zip(col_entropy, probs):
            norm += ent * p
        print (info_gain(total_ent, norm))
      
if(__name__ == "__main__"):
    """
    Main program
    """
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
    # normalised_data = (df - df.min())/(df.max() -  df.min())
    # print "Data after Min Max Normalisation"
    # print (normalised_data.head())
    # # Z-score Normalisation
    # z_normalised_data = (df - df.mean())/df.std()
    # print "Data after Z-score Normalisation"
    # print (z_normalised_data.head())

    # # Equal Width
    info = sonar_data.describe()
    print (type(info))
    # bins = equal_width(df,info,10)
    # for col in df:
    #     hist = sns.distplot(df['Band1'], bins=10)
    #     plt.figure()
    # Plot Boxplot for all attributes
    # ax = sns.boxplot(data=normalised_data)
    # ax = sns.swarmplot(data=normalised_data)

    #information gain
    result = equal_width(df,info, 3)
    types = sonar_data['Type']
    ig = information_gain(df, types)
    print (ig)
    plt.show()       