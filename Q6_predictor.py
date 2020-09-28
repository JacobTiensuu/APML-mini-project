# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Q5_ADF 

def predictor(mu_start, var_start, var_t, num_samples, burn_in, shuffle, extension):
    pred_true, pred_false = Q5_ADF.ADF(mu_start, var_start, var_t,num_samples, burn_in, shuffle, extension)
    r = pred_true/(pred_true + pred_false)
    print(("\nRatio of correctly predicted games:"))
    print(r)
    print("\n")
    #flaw: player one will be predicted to win at first since both players have the same mean skill

def main():
    mu_start = 10
    var_start = mu_start/3
    var_t = 1
    y = 1
    num_samples = 200
    burn_in= 25
    shuffle=1
    extension=0
    predictor(mu_start, var_start, var_t, num_samples, burn_in, shuffle, extension)

if __name__ == "__main__":
    main()