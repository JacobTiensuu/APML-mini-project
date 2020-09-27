# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gibbs_sampler(mu_1, mu_2, sigma_1, sigma_2, sigma_t, y, num_samples, burn_in):
    # Number of samples k
    k = num_samples
    
    # Storing vectors
    s_1 = np.zeros(burn_in+k)
    s_2 = np.zeros(burn_in+k)
    out = np.zeros(burn_in+k)
    
    # Set initial values
    s_1[0] = np.random.normal(mu_1, sigma_1)
    s_2[0] = np.random.normal(mu_2, sigma_2) 
    
    for i in range(burn_in+k-1):
        mu_t = s_1[i] - s_2[i]
        if(y == 1):
            a = (0-(mu_t))/sigma_t
            b = np.infty
        else:
            a = -np.infty
            b = (0-(mu_t))/sigma_t
        # Calculate the output   
        out[i+1] = stats.truncnorm.rvs(a,b,mu_t,sigma_t)
        # Get sigma and Mu from the posterior to calculate the new prior
        Sigma = 1/(sigma_1+sigma_2+sigma_t)*np.matrix([[sigma_1*(sigma_2+sigma_t), sigma_1*sigma_2], \
                                                       [sigma_1*sigma_2, sigma_2*(sigma_1+sigma_t)]])
        Mu = np.matmul(Sigma, np.matrix([[mu_1/sigma_1+out[i+1]/sigma_t], [mu_2/sigma_2-out[i+1]/sigma_t]]))
        # Generate the new skills
        s_1[i+1], s_2[i+1] = np.random.multivariate_normal((Mu[0,0], Mu[1,0]), Sigma)
    
    # Discard burn-in samples
    s_1 = s_1[burn_in:-1]
    s_2 = s_2[burn_in:-1]
    mu_1 = np.mean(s_1)
    mu_2 = np.mean(s_2)
    var_1 = np.var(s_1)
    var_2 = np.var(s_2)
    
    return s_1, s_2, mu_1, mu_2, var_1, var_2

# Calculate

def main():
    # Test the gibbs sampler
    mu_1 = 1
    mu_2 = 1
    sigma_1 = 1
    sigma_2 = 4
    sigma_t = 5
    y = 1
    num_samples = 200
    burn_in = 0
    s1,s2, mu_1, mu_2, var_1, var_2 = gibbs_sampler(mu_1, mu_2, sigma_1, sigma_2, sigma_t, y, num_samples, burn_in)

    # Plot figure
    plt.figure(1)
    plt.plot(s1)
    plt.show()  

if __name__ == "__main__":
    main()


        