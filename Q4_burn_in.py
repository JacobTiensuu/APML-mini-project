# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Q4_gibbsSampler

def burn_in(mu_1, mu_2, var_1, var_2, var_t,y,num_samples):
    burn_in = 0

    # Illustrating the burn in-period by plotting 10 different posterior samplings
    # When they all seem to have reached stationarity we know how many samples we should throw away to get good approximations
    for i in range(10):
        s1,s2, mu_1, mu_2, var_1, var_2 = Q4_gibbsSampler.gibbs_sampler(mu_1, mu_2, var_1, var_2, var_t, y, num_samples, burn_in)
        plt.plot(s1)
        #plt.plot(s2)
        # Plot figure
    plt.title("Burn in illustration")
    plt.xlabel("Sample #")
    plt.ylabel("Sample value")
    plt.show()

def main():
    mu_start = 20
    var_start = 5
    var_t = 1
    y = 1
    num_samples = 100
    burn_in(mu_start, mu_start, var_start, var_start, var_t,y,num_samples)

if __name__ == "__main__":
    main()
