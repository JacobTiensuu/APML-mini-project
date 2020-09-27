# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Q4_gibbsSampler
import Q4_burn_in
import Q5_ADF
import Q6_predictor


def main():
    
    # value = float(input("Enter a float for mean value:\n"))
    # print(f'You entered {value} and its square is {value ** 2}')

    #Define values to be used
    mu_start = 10
    var_start = mu_start/3
    var_t = 1
    y = 1
    num_samples = 200
    burn_in= 25
    shuffle=1

    #Q 4
        #Testing the sampler
    s1,s2, mu_1, mu_2, var_1, var_2 = Q4_gibbsSampler.gibbs_sampler(mu_start, mu_start, var_start, var_start, var_t, y, num_samples, burn_in)
    # Plot figure
    plt.figure(1)
    plt.plot(s1)
    plt.show()

        #Testing the burn in
    Q4_burn_in.burn_in(mu_start, mu_start, var_start, var_start, var_t,y,num_samples)
      

    #Q 5
    Q5_ADF.ADF(mu_start, var_start, var_t, num_samples, burn_in, shuffle)

    #Q 6
    Q6_predictor.predictor(mu_start, var_start, var_t,num_samples, burn_in, shuffle)


if __name__ == "__main__":
    main()