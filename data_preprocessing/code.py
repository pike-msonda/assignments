import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

plt.style.use(['bmh'])
FILENAME = "sonar.dat"
CLASS_NAME = "Type"
#TODO: Streamline the code to be more efficient 
def mode(data):
    mode_index_size =  data.index.size
    for m in data:
        if data[m].isnull().any():
            data[m] = data[m].fillna(0).sum()
        else:
            data[m] = data[m].sum()/ mode_index_size
    return data.head(1)

def random_colors():
    """
        Generate random color index from given index set
        color_index: An array of colors to be displayed.
    """
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color

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

def equal_width(data,bins):
    '''
        Basic Binning Method. Splits the attribute into n bins based on intervals. 

        data : pandas.DataFrame
        info : pandas.DataFrame
        bins : Integer
    '''
    output_frame = pd.DataFrame()
    for column in data:
        output_frame[column] = pd.cut(data[column], bins,labels=create_labels(bins), retbins=False)
    return output_frame

def remove_classes(data):
    '''
        Remove the class column fromt pandas.DataFrame

        data: pandas.DataFrame
        
        class_name: String
    '''
    classes  = data[CLASS_NAME]

    return data.drop([CLASS_NAME], axis = 1), classes

def create_labels(bins):
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
    data[CLASS_NAME] = types
    size= data.index.size
    cases = data.groupby(CLASS_NAME).count().values
    n_rcases = float(cases.min())
    n_mcases = float(cases.max())
    total_prob = [n_mcases/size, n_rcases/size]
    total_ent = entropy(total_prob, base= 2)
    data = remove_classes(data)[0] # Remove the column class

    for col in data:
        counts =  data[col].value_counts().values
        counts = [float(x) for x in counts]
        probs = [x / size for x in counts]
        attribute_entropy = get_entropy(cond_P(data[col],types))
        info_gain = total_ent - sum([a * b for a,b in zip(attribute_entropy, probs)])
        gain.append({col: info_gain})

    return gain
def load_data(filename):
    # Read from sonar.dat file
    data = pd.read_table(filename,sep=',',usecols=[0,1,2,3,60]) #slect the frist 4 columns and the class column

    return data

def main():

    df, types = remove_classes(load_data(FILENAME)) # Temporarily remove the type column
    equalWidthIndex = [3,4]     #equal width binning  
    #5 Row Summary:
    print ("First 5 rows Summary:")
    print (load_data(FILENAME).head().to_csv("summary.csv"))
    
    # Mean, Mode, Standard Deviation, Variance
    print ("Mean, Standard Deviation, InterQuartile, Min, Max, Count:")
    print (df.agg(['mean','std','var','quantile']))

    # Mode 
    print ("Mode:")
    mde = df.mode()
    md_res= mode(mde)
    md_res.to_csv("mode.csv")
    print(md_res)

    normalised_data = min_max_normalization(df)
    print ("Data after Min Max Normalisation:")
    print (normalised_data.head())
    normalised_data.round(3).to_csv("normaliseddata.csv")
    z_normalised_data = z_normalization(df)
    print ("Data after Z-score Normalisation:")
    print (z_normalised_data.head())
    z_normalised_data.round(3).to_csv("z-normalieddata.csv")

    # # Frequency Histogram 
    #TODO: add names to graphs and assign colors randomly.
    index = 221
    for col in df:
        plt.subplot(index)
        ax = sns.distplot(df[col],kde=True, color=random_colors())
        index += 1

    #Plot Boxplot for all attributes
    plt.suptitle('Histograms Showing the Frequency in given Attributes')
    plt.figure()
    ax = sns.boxplot(data=df)
    ax.set_title("Box Plot Attributes")
    #ax = sns.swarmplot(data=df)

    #information gain
    for n in equalWidthIndex:
        eqw= equal_width(df, n)
        info_gain = information_gain(eqw, types)
        print ("Information Gain using {} Equal -Width method".format(n))
        for gain in info_gain:
            print (gain)     

   
if(__name__ == "__main__"):
    '''
        Data Processing Assignment to perform pre-processing operations on a given data set {sonar.dat}

        Data Description included in the file names.txt

        Main program
    '''
    main()
    plt.show()