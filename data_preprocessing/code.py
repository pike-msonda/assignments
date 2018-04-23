import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy


#TODO: Add exception handling
#TODO: Streamline the code to be more efficient 
def min_max_normalization(data):
    return (data - data.min())/(data.max() -  data.min())

def equal_width(data, info,bins):
    '''
        Basic Binning Method. Splits the attribute into n bins based on intervals. 

        data : pandas.DataFrame
        info : pandas.DataFrame
        bins : Integer
    '''

    for column in info:
        data[column] = pd.cut(data[column], bins,labels=create_label(bins))
    return data

def remove_classes(data):
    '''
        Remove the class column fromt pandas.DataFrame

        data: pandas.DataFrame
    '''

    return data.drop(['Type'], axis = 1)

def create_label(bins):
    '''
        Simple Function to create labels correspnding to the numbers of bins requested
        bins: Integer
    '''

    labels = []
    for x in range(0, bins):
        labels.append("bin"+str(x))
    return labels 


def cond_P(A,B):
    '''
        Get Probabilities between two attributes. P(A|B)

        A: pandas.DataFrame.column
        B: pandas.DataFrame.column
    '''

    cond_events = A.groupby(B, sort=False)
    values = cond_events.value_counts(normalize=False).unstack(level=1).fillna(0)
    prob = []
    for col in values:
        total = sum(values[col])
        r = float(values[col][0])
        m = float(values[col][1])
        prob.append([np.divide(r,total), np.divide(m,total)])
    return prob

def get_entropy(A):
    '''
        Calculate the entropy of each attribute value
        
        A: A list of probabilities. 

    '''

    entropy_list = []  # list to store all the entropy 
    for a in A:
        if 0 in a: #Check if the probability has 0, which can't be computed by log2
            ix = a.index(0)
            a[ix] = 1
        entropy_list.append(entropy(a, base=2))
    return entropy_list

def information_gain(data,types):
    ''' 
        Function to calcuate the information gain of the given data

        data : given data set. Pandas.DataFrame

        types: data classes.

        Mutual Information of Information Gain: Entropy(S) - Entropy(P|A)

    '''
    #TODO: Cut down the number of unnecessary operations
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
        attribute_entropy = get_entropy(cond_P(data[col],types))
        info_gain = total_ent - sum([a * b for a,b in zip(attribute_entropy, probs)])
        gain.append({col: info_gain})

    return gain

'''
    Data Processing Assignment to perform pre-processing operations on a given data set {sonar.dat}
    Data Description included in the file names.txt
'''     
if(__name__ == "__main__"):
    '''
    Main program
    '''

    # Read from sonar.dat file
    sonar_data =  pd.read_table('sonar.dat',sep=',',usecols=[0,1,2,3,60])
    df = remove_classes(sonar_data) # Temporarily remove the type column
    types = sonar_data['Type']
    n = 3
    #5 Row Summary:
    print (sonar_data.head())
    
    # Mean, Mode, Standard Deviation, Variance
    print (df.agg(['mean','std','var','min', 'max', 'count','quantile',{'min_max_norm': min_max_normalization }]))

    
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
    eqw= equal_width(df,info, n)
    info_gain = information_gain(eqw, types)
    print ("Information Gain using {} Equal -Width method".format(n))
    print (info_gain)     