import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

#TODO: Streamline the code to be more efficient 
#TODO: Add mode function to aggregrates

def min_max_normalization(data):
    '''
        Calculate the min-max Normalization for a dataset.

        data: pandas.DataFrame
    '''
    return (data - data.min())/(data.max() -  data.min())

def z_normalization(data):
    '''
        Calculate the z-Normalization for a dataset.

        data: pandas.DataFrame
    '''
    return (data - data.mean())/data.std()

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

    cond_events = A.groupby(B, sort=False).value_counts(normalize=False).unstack(level=1).fillna(0)
    prob = []
    for col in cond_events:
        total = sum(cond_events[col])   # get total items for each bin 
        r = float(cond_events[col][0])  # total class 'R' items
        m = float(cond_events[col][1])  # total class 'M' items 
        prob.append([np.divide(r,total), np.divide(m,total)])
    return prob

def get_entropy(A):
    '''
        Calculate the entropy of each attribute value
        
        A: A list of probabilities. <list>
    '''
    try:
        entropy_list = []  # list to store all the entropy 
        for a in A:
            entropy_list.append(entropy(a, base=2))

    except ZeroDivisionError:
        print ("An error occurred, this {} list has zero which can't be computed by log2".format(a))

    return entropy_list

def information_gain(data,types):
    ''' 
        Function to calcuate the information gain of the given data

        data : given data set. Pandas.DataFrame

        types: data classes. <list>

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
    n = 3       #equal width binning  
    #5 Row Summary:
    print ("First 5 rows Summary:")
    print (sonar_data.head())
    
    # Mean, Mode, Standard Deviation, Variance
    print ("Mean, Mode, Standard Deviation, InterQuartile, Min, Max, Count:")
    print (df.agg(['mean','std','var','min', 'max', 'count','quantile']))

    
    normalised_data = min_max_normalization(df)
    print ("Data after Min Max Normalisation:")
    print (normalised_data.head())
    # Z-score Normalisation
    z_normalised_data = z_normalization(df)
    print ("Data after Z-score Normalisation:")
    print (z_normalised_data.head())

    # # Frequency Histogram 
    #TODO: add names to graphs and assign colors randomly.
    info = sonar_data.describe()
    for col in df:
        clr_palette = sns.color_palette()
        hist = sns.distplot(df[col], bins=10,kde=False, color="red")
        plt.figure()

    #Plot Boxplot for all attributes
    ax = sns.boxplot(data=normalised_data)
    ax = sns.swarmplot(data=normalised_data)

    #information gain
    eqw= equal_width(df,info, n)
    info_gain = information_gain(eqw, types)
    print ("Information Gain using {} Equal -Width method".format(n))
    print (info_gain)     
    plt.show()