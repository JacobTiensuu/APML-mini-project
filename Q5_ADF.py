# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Q4_gibbsSampler import gibbs_sampler as gibbs

def ADF():

    # Read data from csv file and remove date and time stamp and draws
    data = pd.read_csv('SerieA.csv')
    data.drop(labels = ['yyyy-mm-dd', 'HH:MM'], axis = 1)

    i = np.where(data['score1'] == data['score2'])
    print(i)


ADF()
    
#def main():
#    ADF()


#def if __name__ == "__main__":
#    main()




