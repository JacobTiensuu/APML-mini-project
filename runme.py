# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Q4_gibbsSampler
import Q4_burn_in
import Q5_ADF
import Q6_predictor
import Q8
import Q9_mydata

def main():
    #Define parameter values to be used
    mu_start = 10           #The mean skill all teams will start with
    var_start = mu_start/3  #Variance of starting skill
    var_t = 1               #Variance in function t
    y = 1                   #Which player wins the game for single game questions, e.g Q4
    num_samples = 100       #Number of samples the Gibbs-sampler will use
    burn_in = 25            #Burn in period/samples to be discarded
    shuffle = 1             #Set to 1 to shuffle the data for the matches. Other values turns it off
    extension = 1           #Set to 1 to use the home advantage extension in Q5

    # #Q 4
        #Testing the sampler
    print("Q4: Showing sampled posterior values of player 1's skill after winning a game against player with the same prior skill")
    print("Close graph to continue..\n")
    s1,s2, mu_1, mu_2, var_1, var_2 = Q4_gibbsSampler.gibbs_sampler(mu_start, mu_start, var_start, var_start, var_t, y, num_samples, burn_in)
    # Plot figure
    plt.figure(1)
    plt.plot(s1)
    plt.show()

    #     #Testing the burn in
    print("Q4: Showing 10 runs of samples values without burn-in cut away. No clear burn-in period can be seen")
    print("Close graph to continue..\n")
    Q4_burn_in.burn_in(mu_start, mu_start, var_start, var_start, var_t,y,num_samples) 
    
    #Q 5 and Q6
    # Set home_team_adv to zero for non extened project
    print("Q5 & Q6: Predicting outcomes for the whole season...")
    print("Will soon display final ranking of teams. Close graph to continue.\n")
    Q6_predictor.predictor(mu_start, var_start, var_t,num_samples, burn_in, shuffle, extension)

    #Q 8
    print("Q8: Comparing game outcome calculated with Gibbs-sampling vs moment matching")
    print("Close graph to continue..\n")
    Q8.MomentMatching(mu_start,var_start,var_t,y,num_samples,burn_in)
    
    #Q9
    print("Q9: Testing ADF on NHL-data set..")
    print("Close graph to finish.\n")
    Q9_mydata.ADF(mu_start, var_start, var_t, num_samples, burn_in, shuffle)
    

if __name__ == "__main__":
    main()