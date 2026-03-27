import pandas as pd
from oggm import cfg
import numpy as np
import matplotlib.pyplot as plt
import os

num_of_glaciers = 1
N = 10
cfg.PATHS['working_dir'] = "~/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/glacier_outs"
rgi_ids = ['RGI60-14.00063']

# compile CSVs to plot timeseries and plot to view the mass balance time series for each of the 50 samples, to see how they are looking and check that they make sense before we calculate the sensitivity indices
mass_balance_dict = {}
years_dict = {}
runoff_dict = {}
area_dict = {}
volume_dict = {}
params_dict = {}

for j in range(num_of_glaciers):

    mass_balance_samples = []
    years = []
    runoff_samples = []
    area_samples = []
    volume_samples = []
    params = []

    for i in range(N):
        df = pd.read_csv(cfg.PATHS['working_dir'] + '/' + str(i) + '_' + str(j) +'_pakistan_runoff_output.csv')
        mass_balance_samples.append(df['mass_balance'].values)
        years.append(df['years'].values)
        runoff_samples.append(df['runoff_Mt'].values)
        area_samples.append(df['area_km2'].values)
        volume_samples.append(df['volume_km3'].values)

        mb_param_df = pd.read_csv(cfg.PATHS['working_dir'] + '/' + str(i) + '_' + str(j) +'_pakistan_params.csv') 
        params.append(mb_param_df['params'])

    N = len(params)
    print(N)
    mass_balance_dict[j] = mass_balance_samples
    years_dict[j] = years
    runoff_dict[j] = runoff_samples
    area_dict[j] = area_samples
    volume_dict[j] = volume_samples
    params_dict[j] = np.vstack([p.values for p in params])

    print("done glacier: ", j)

# compile csvs to plot timeseries and plot to view the mass balance time series for each of the 50 samples, to see how they are looking and check that they make sense before we calculate the sensitivity indices
plt.figure(figsize=(13,num_of_glaciers*4))
for j in range(num_of_glaciers):
    for i in range(N):
        plt.subplot(num_of_glaciers,1,j+1)
        plt.plot(years[i], mass_balance_dict[j][i], label='sim', color='k', linewidth=0.5)

    plt.title('Mass balance time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_ids[j]))
    plt.xlabel('Year'), plt.ylabel('Mass Balance kg m$^-2$')
    plt.tight_layout()

    
    outpath = os.path.join(cfg.PATHS['working_dir'], "mass_balance.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

