# Libraries
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Q4_gibbsSampler

def ADF(mu_start, var_start, var_t, num_samples, burn_in, shuffle):

    # Read data from csv file and remove date and time stamp and draws
    data = pd.read_csv('game.csv')
    i = np.where(data['season'] != 20182019)
    data = data.drop(data.index[i])

    i = np.where(data['home_goals'] == data['away_goals'])
    data = data.drop(labels = ['season', 'type', 'date_time', 'date_time_GMT', 'outcome'], axis = 1)
    data = data.drop(data.index[i])

    team_info = pd.read_csv('team_info.csv')
    team_info = team_info.drop(labels = ['franchiseId', 'teamName', 'abbreviation', 'link'], axis = 1)
    
    team1 = data.home_team_id.unique()
    skills = np.ones(len(team1))*mu_start
    vars = np.ones(len(team1))*var_start

    # Create dataframe to represent team 
    teams = {'Name' : team1,
            'skill' : skills,
            'variance': vars,
            'rank': skills-3*var_start}

    teams = pd.DataFrame(teams)
    
    #Define confusion matrix
    pred_true = 0
    pred_false = 0

    for i in (data.index):
        team1, team2 = data.loc[i, 'away_team_id'], data.loc[i,'home_team_id']
        mu_1 = teams.loc[teams['Name'] == team1, 'skill'].iat[0]
        var_1 = teams.loc[teams['Name'] == team1, 'variance'].iat[0]
        mu_2 = teams.loc[teams['Name'] == team2, 'skill'].iat[0]
        var_2 = teams.loc[teams['Name'] == team2, 'variance'].iat[0]
        t = data.loc[i, 'away_goals'] - data.loc[i, 'home_goals']
        y = np.sign(t)

        #Check predictions for Q6
        p = np.sign(mu_1-mu_2) #for Q6
        if(p == y):
            pred_true = pred_true + 1
        else:
            pred_false = pred_false + 1

        s_1, s_2, mu_1, mu_2, var_1, var_2 = Q4_gibbsSampler.gibbs_sampler(mu_1, mu_2, var_1, var_2, var_t, y, num_samples, burn_in)

        teams.at[teams['Name']==team1,'skill'] = mu_1 
        teams.at[teams['Name']==team1,'variance'] = var_1 
        teams.at[teams['Name']==team2,'skill'] = mu_2 
        teams.at[teams['Name']==team2,'variance'] = var_2
        teams.at[teams['Name']==team1, 'rank'] = mu_1-3*var_1
        teams.at[teams['Name']==team2, 'rank'] = mu_2-3*var_2
        
    # Sorting the dataframe
    teams = teams.sort_values(by='rank', ascending=False)
    
    # Plotting data
    Name = teams['Name'].to_numpy()
    rank = teams['rank'].to_numpy()
    team_id = team_info['team_id'].to_numpy()
    short_name = team_info['shortName'].to_numpy()
    team_name = np.empty(len(Name), dtype = 'object')
    
    for i in range(len(Name)):
        for j in range(len(team_id)):
            if(Name[i] == team_id[j]):
                team_name[i] = short_name[j]

    fig, ax = plt.subplots()
    y_pos = np.arange(len(team_name))
    ax.barh(y_pos, rank, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(team_name)
    ax.invert_yaxis()
    ax.set_xlabel('Ranking')
    ax.set_title('Chart at the end of the season')
    plt.show()

    return pred_true, pred_false

def main():
    mu_start = 20
    var_start = 20/3
    var_t = 1.5
    num_samples = 500
    burn_in = 25
    shuffle = 0
    ADF(mu_start, var_start, var_t, num_samples, burn_in, shuffle)

if __name__ == "__main__":
    main()