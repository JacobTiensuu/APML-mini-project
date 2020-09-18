# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Q5_ADF 

def predictor():
    mu_start = 10
    sigma_start = 10/3
    sigma_t = 1 
    num_samples = 300
    burn_in = 100
    pred_true, pred_false = Q5_ADF.ADF(mu_start, sigma_start, sigma_t,num_samples, burn_in)
    print(pred_true)
    print(pred_false)
    #flaw: player one will be predicted to win at first since both players have the same mean skill



def main():
    predictor()

if __name__ == "__main__":
    main()