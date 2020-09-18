# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Q4_gibbsSampler

def ADF(mu_start, sigma_start, sigma_t, num_samples, burn_in):

    # Read data from csv file and remove date and time stamp and draws
    data = pd.read_csv('SerieA.csv')
    i = np.where(data['score1'] == data['score2'])

    data = data.drop(labels = ['yyyy-mm-dd', 'HH:MM'], axis = 1)
    data = data.drop(data.index[i])
    

    team1 = data.team1.unique()
    skills = np.ones(len(team1))*mu_start
    vars = np.ones(len(team1))*sigma_start

    # Create dataframe to represent team 
    teams = {'Name' : team1,
            'skill' : skills,
            'variance': vars,
            'rank': skills-3*sigma_start}

    teams = pd.DataFrame(teams)
    
    #Define confusion matrix
    pred_true = 0
    pred_false = 0

    for i in (data.index):
        team1, team2 = data.loc[i, 'team1'], data.loc[i,'team2']
        mu_1 = teams.loc[teams['Name'] == team1, 'skill'].iat[0]
        sigma_1 = teams.loc[teams['Name'] == team1, 'variance'].iat[0]
        mu_2 = teams.loc[teams['Name'] == team2, 'skill'].iat[0]
        sigma_2 = teams.loc[teams['Name'] == team2, 'variance'].iat[0]
        t = data.loc[i, 'score1'] - data.loc[i, 'score2']
        p = sign(mu_1-mu_2) #for Q6
        y = np.sign(t)

        #Check predictions for Q6
        if(p == y):
            pred_true = pred_true + 1
        else:
            pred_false = pred_false + 1

        s_1, s_2, mu_1, mu_2, sigma_1, sigma_2 = Q4_gibbsSampler.gibbs_sampler(mu_1, mu_2, sigma_1, sigma_2, sigma_t, y, num_samples, burn_in)

        teams.at[teams['Name']==team1,'skill'] = mu_1 
        teams.at[teams['Name']==team1,'variance'] = sigma_1 
        teams.at[teams['Name']==team2,'skill'] = mu_2 
        teams.at[teams['Name']==team2,'variance'] = sigma_2
        teams.at[teams['Name']==team1, 'rank'] = mu_1-3*sigma_1
        teams.at[teams['Name']==team2, 'rank'] = mu_2-3*sigma_2


    # Sorting the dataframe
    teams = teams.sort_values(by='rank', ascending=False)
    print(teams)
    return pred_true, pred_false

def main():
    mu_start = 10
    sigma_start = 10/3
    sigma_t = 1 
    num_samples = 300
    burn_in = 100
    ADF(mu_start, sigma_start, sigma_t, num_samples, burn_in)


if __name__ == "__main__":
    main()




