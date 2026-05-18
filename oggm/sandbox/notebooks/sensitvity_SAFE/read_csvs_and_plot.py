import pandas as pd
from oggm import cfg
import numpy as np
import matplotlib.pyplot as plt
import os
import oggm
from oggm import cfg, workflow, utils
from oggm.core.sensitivity import hydro_output_metric_calculator

import oggm.core.sensitivity
import importlib
from oggm.core.sensitivity import parameter_bounding


import safepython.PAWN as PAWN # Module to calculate PAWN sensitivity indices
from safepython.sampling import AAT_sampling # Functions to perform the input sampling
from safepython.util import aggregate_boot # Functions to perform bootstrapping
import safepython.plot_functions as pf

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Run OGGM glacier sensitivity analysis"
    )

    parser.add_argument("--work_dir", type=str, required=True,
                        help="Path to OGGM working directory")

    parser.add_argument("--rgi_ids", nargs='+', required=True,
                        help="RGI IDs to process (space-separated or single string)")

    parser.add_argument("--N", type=int, required=True,
                        help="Number of samples")

    parser.add_argument("--output_csv_path", type=str, required=True,
                        help="Output CSV paths")
    
    parser.add_argument("--params_csv_path", type=str, required=True,
                        help="Parameters CSV paths")

    parser.add_argument("--x_max", type=str, required=True,
                        help="Maximum values for each parameter (space-separated)")

    parser.add_argument("--x_min", type=str, required=True,
                        help="Minimum values for each parameter (space-separated)")


    return parser.parse_args()

def main():

    # Initialising
    args = get_args()
    cfg.initialize(logging_level='CRITICAL')
    cfg.PATHS['working_dir'] = args.work_dir
    cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['min_ice_thick_for_length'] = 1
    cfg.PARAMS['use_multiprocessing'] = True  # To speed up sensitivity analysis run
    
    # This is a list of RGI IDs
    rgi_ids = args.rgi_ids
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup'
    gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=4, prepro_border=160, prepro_base_url=base_url)
    
    num_of_glaciers = len(rgi_ids)
    N = args.N

    rgi_dates = []
    rgi_area_km2s = []

    for gdir in gdirs:
        rgi_dates.append(gdir.rgi_date)
        rgi_area_km2s.append(gdir.rgi_area_km2)

    # And match the Hugonnet
    geo_df = utils.get_geodetic_mb_dataframe()

    mask = geo_df['period'].eq('2000-01-01_2020-01-01')
    selected_gdirs_geo_df = geo_df.loc[geo_df.index.isin([str(rgi_ids[0])]) & mask]

    hugonnet_dmdt = selected_gdirs_geo_df['dmdtda'].values * rgi_area_km2s

    hugonnet_err_dmdt = selected_gdirs_geo_df['err_dmdtda'].values * rgi_area_km2s
    
    ##############################################################
    # Now call all of our functions!
    ##############################################################
    mass_balance_dict, years_dict, runoff_dict, area_dict, volume_dict, params_dict, successful_params_dict = read_csvs(args, num_of_glaciers, gdirs, N)

    # N = len(mass_balance_dict[0]) # Update N to reflect the number of successful simulations (this is important for the plotting functions and the sensitivity analysis, as we want to make sure we are only including the successful simulations in these steps, and not the ones that failed or are missing)
    plot_mass_balance_timeseries(mass_balance_dict, years_dict, rgi_ids, num_of_glaciers, N)
    plot_area_timeseries(area_dict, years_dict, rgi_ids, num_of_glaciers, N, gdirs)
    plot_volume_timeseries(volume_dict, years_dict, rgi_ids, num_of_glaciers, N)
    plot_runoff_timeseries(runoff_dict, years_dict, rgi_ids, num_of_glaciers, N)
    plot_input_hists(params_dict, args)
    plot_input_distributions(num_of_glaciers, N, mass_balance_dict, rgi_ids)

    YY_dict = hydro_output_calculator(runoff_dict, num_of_glaciers)
    plot_mean_vs_std_runoff(YY_dict, num_of_glaciers)
    plot_runoff_mean_and_std_distributions(YY_dict, num_of_glaciers)
    runoff_pawn_plot_all(successful_params_dict, YY_dict, metric='Mean Runoff', n=5, Nboot=500, k=0)

    plot_parameter_bounding(params_dict, successful_params_dict, area_dict, mass_balance_dict, hugonnet_dmdt, hugonnet_err_dmdt, rgi_area_km2s, rgi_dates, years_dict, num_of_glaciers, N)


##############################################################
# Read CSVs
##############################################################

def read_csvs(args, num_of_glaciers, gdirs, N):
    # Compile the CSVs 
    mass_balance_dict = {}
    years_dict = {}
    runoff_dict = {}
    area_dict = {}
    volume_dict = {}
    params_dict = {}
    successful_params_dict = {}
    for j in range(num_of_glaciers):
        mass_balance_samples = []
        years_samples = []
        runoff_samples = []
        area_samples = []
        volume_samples = []
        params = []
        successful_params = []
        for i in range(N):
            try:
                path = cfg.PATHS['working_dir'] + '/' + str(i) + args.output_csv_path
                df = pd.read_csv(path)
                # Append output arrays
                area_samples.append(df['area_km2'].values[1:])
                runoff_samples.append(df['runoff_Mt'].values[1:])
                mass_balance_samples.append(df['mass_balance'].values[1:])
                volume_samples.append(df['volume_km3'].values[1:])
                years_samples.append(df['years'].values[1:])
                
                # Read parameters for successful simulations
                mb_param_df = pd.read_csv(
                    cfg.PATHS['working_dir'] + '/' + str(i) + args.params_csv_path
                )
                successful_params.append(mb_param_df['params'].to_numpy().ravel())
            except FileNotFoundError:
                print(f"No simulation for index {i}")
            try:
                # Read parameters
                mb_param_df = pd.read_csv(
                    cfg.PATHS['working_dir'] + '/' + str(i) + args.params_csv_path
                )
                params.append(mb_param_df['params'].to_numpy().ravel())
            except FileNotFoundError:
                print(f"No parameter file for index {i}")

        mass_balance_dict[j] = mass_balance_samples
        years_dict[j] = years_samples
        runoff_dict[j] = runoff_samples
        area_dict[j] = area_samples
        volume_dict[j] = volume_samples
        params_dict[j] = np.vstack(params)
        successful_params_dict[j] = np.vstack(successful_params)
        print("done glacier: ", gdirs[j].rgi_id)
    return mass_balance_dict, years_dict, runoff_dict, area_dict, volume_dict, params_dict, successful_params_dict

##############################################################
# Plotting Simulation Outputs
##############################################################
def plot_mass_balance_timeseries(mass_balance_dict, years_dict, rgi_ids, num_of_glaciers, N):
    # compile csvs to plot timeseries and plot to view the mass balance time series for each of the samples, to see how they are looking and check that they make sense before we calculate the sensitivity indices
    plt.figure(figsize=(13,num_of_glaciers*4))
    print(len(years_dict[0]))
    for j in range(num_of_glaciers):
        for i in range(len(years_dict[j])):
            plt.subplot(num_of_glaciers,1,j+1)
            plt.plot(years_dict[j][i], mass_balance_dict[j][i], label='sim', color='k', linewidth=0.5)
        plt.title('Mass balance time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_ids[j]))
        plt.xlabel('Year'), plt.ylabel('Mass Balance kg m$^-2$')
        plt.tight_layout()
        outpath = os.path.join(cfg.PATHS['working_dir'], "mass_balance.png")
        plt.savefig(outpath, dpi=200, bbox_inches='tight')

def plot_area_timeseries(area_dict, years_dict, rgi_ids, num_of_glaciers, N, gdirs):
    plt.figure(figsize=(13,num_of_glaciers*4))
    for j in range(num_of_glaciers):
        for i in range(len(years_dict[j])):
            plt.subplot(num_of_glaciers,1,j+1)
            plt.plot(years_dict[j][i], area_dict[j][i], label='sim', color='k', linewidth=0.5)
        plt.scatter(gdirs[j].rgi_date, gdirs[j].rgi_area_km2, color='r', s=20, zorder=999)
        plt.title('Area time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_ids[j]))
        plt.ylabel('Glacier area (km$^2$)')
        plt.xlabel('Year')
        plt.tight_layout()
        outpath = os.path.join(cfg.PATHS['working_dir'], "area.png")
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.figure(figsize=(13,num_of_glaciers*4))

def plot_volume_timeseries(volume_dict, years_dict, rgi_ids, num_of_glaciers, N):
    plt.figure(figsize=(13,num_of_glaciers*4))
    for j in range(num_of_glaciers):
        for i in range(len(years_dict[j])):
            plt.subplot(num_of_glaciers,1,j+1)
            plt.plot(years_dict[j][i], volume_dict[j][i], label='sim', color='k', linewidth=0.5)
        plt.title('Volume time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_ids[j]))
        plt.ylabel('Glacier volume (km$^3$)')
        plt.xlabel('Year')
        plt.tight_layout()
        outpath = os.path.join(cfg.PATHS['working_dir'], "volume.png")
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.figure(figsize=(13,num_of_glaciers*4))

def plot_runoff_timeseries(runoff_dict, years_dict, rgi_ids, num_of_glaciers, N):
    plt.figure(figsize=(13,num_of_glaciers*4))
    for j in range(num_of_glaciers):
        for i in range(len(years_dict[j])):
            plt.subplot(num_of_glaciers,1,j+1)
            plt.plot(years_dict[j][i], runoff_dict[j][i], label='sim', color='k', linewidth=0.5)

        plt.title('Runoff time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_ids[j]))

        plt.ylabel('Runoff (Mt/yr)')
        plt.xlabel('Year')
        plt.tight_layout()
        outpath = os.path.join(cfg.PATHS['working_dir'], "runoff.png")
        plt.savefig(outpath, dpi=200, bbox_inches='tight')

##############################################################
# Plotting distribution of Inputs
##############################################################
def plot_input_hists(params_dict, args):

    x_max = [float(v) for v in args.x_max.split()]
    x_min = [float(v) for v in args.x_min.split()]
    melt_low, melt_high = x_min[0], x_max[0]
    precip_low, precip_high = x_min[1], x_max[1]
    tbias_low, tbias_high = x_min[2], x_max[2]
    nbins = 50
    bins_melt   = np.linspace(melt_low,   melt_high,   nbins+1)
    bins_precip = np.linspace(precip_low, precip_high, nbins+1)
    bins_tbias  = np.linspace(tbias_low,  tbias_high,  nbins+1)
    plt.figure(figsize=(13, 4))
    plt.subplot(1,3,1)
    plt.title('Distribution of melt factor', loc='left')
    plt.hist(params_dict[0][:,0], bins=bins_melt, range=(melt_low, melt_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Melt Factor')
    plt.subplot(1,3,2)
    plt.title('Distribution of precipitation factor', loc='left')
    plt.hist(params_dict[0][:,1], bins=bins_precip, range=(precip_low, precip_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Precipitation Factor')
    plt.subplot(1,3,3)
    plt.title('Distribution of temperature bias', loc='left')
    plt.hist(params_dict[0][:,2], bins=bins_tbias, range=(tbias_low, tbias_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Temperature Bias')
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "parameter_distribution.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

##############################################################
# Plotting Mass Balance Means and Standard Deviations
##############################################################
def plot_input_distributions(num_of_glaciers, N, mass_balance_dict, rgi_ids):
    mean_mass_balances_dict = {}
    std_mass_balances_dict  = {}
    for j in range(num_of_glaciers):     
        mean_mass_balances_dict[j] = []
        std_mass_balances_dict[j]  = []
        for i in range(len(mass_balance_dict[j])): # number of samples
            mb_series = mass_balance_dict[j][i]
            mean_mass_balances_dict[j].append(mb_series.mean())
            std_mass_balances_dict[j].append(mb_series.std())
    plt.figure(figsize=[13,4*num_of_glaciers])
    for j in range(num_of_glaciers):
        min_means, max_means = np.nanmin(mean_mass_balances_dict[j]), np.nanmax(mean_mass_balances_dict[j])
        min_stds, max_stds = np.nanmin(std_mass_balances_dict[j]), np.nanmax(std_mass_balances_dict[j])
        nbins = 100
        bins_means = np.linspace(min_means, max_means, nbins+1)
        bins_stds = np.linspace(min_stds, max_stds, nbins+1)
        plt.subplot(num_of_glaciers,2,2*j+1), 
        plt.title('Distribution of Mean Mass Balance for RGI_ID = %s ' % rgi_ids[j], loc='left')
        plt.hist(mean_mass_balances_dict[j], range=(min_means, max_means), color='grey');
        plt.ylabel('Frequency of samples'), plt.xlabel('Mean')
        # plt.hist(mean_mass_balances_dict[j], bins='auto', range=(min_means, max_means), color='grey');
        # plt.ylabel('Frequency of samples'), plt.xlabel('Mean')
        # plt.subplot(num_of_glaciers,2,2*j+2), plt.title('Distribution of Std of Mass Balance', loc='left'), plt.hist(std_mass_balances_dict[j], bins='auto', 
        #                                                                                          range=(min_stds, max_stds), color='grey');
        plt.subplot(num_of_glaciers,2,2*j+2), plt.title('Distribution of Std of Mass Balance', loc='left'), plt.hist(std_mass_balances_dict[j], bins='auto', color='grey');
        plt.ylabel('Frequency of samples'), plt.xlabel('Std')
        plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "output_distributions.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

def hydro_output_calculator(runoff_dict, num_of_glaciers):
    YY_dict = {}
    for j in range(num_of_glaciers):   
        YY_dict[j] = hydro_output_metric_calculator(runoff_dict[j])
    return YY_dict
##############################################################
# Plot of Mean vs Std Runoff values
##############################################################
def plot_mean_vs_std_runoff(YY_dict, num_of_glaciers):
    plt.figure(figsize=[13,num_of_glaciers*4])
    for j in range(num_of_glaciers):  
        plt.subplot(num_of_glaciers,3,j+1), plt.scatter(YY_dict[j][:,0], YY_dict[j][:,1], s=5), plt.title('Mean runoff vs Std of runoff'), plt.xlabel('Mean runoff Mt/yr'), plt.ylabel('Std of runoff Mt/yr'),
        plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "mean_vs_std_runoff.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
##############################################################
# Histogram plot of Runoff Mean and Std
##############################################################
def plot_runoff_mean_and_std_distributions(YY_dict, num_of_glaciers):

    plt.figure(figsize=[13,num_of_glaciers*4])
    for j in range(num_of_glaciers):
        min_mean_ro, max_mean_ro = np.min(YY_dict[j][:,0]), np.max(YY_dict[j][:,0])
        min_std_ro, max_std_ro = np.min(YY_dict[j][:,1]), np.max(YY_dict[j][:,1])
        nbins = 50
        bins_mean_ro = np.linspace(min_mean_ro, max_mean_ro, nbins+1)
        bins_std_ro = np.linspace(min_std_ro, max_std_ro, nbins+1)
        plt.subplot(num_of_glaciers,2, 2*j+1), plt.title('Distribution of Runoff Mean', loc='left'), 
        # plt.hist(YY_dict[j][:,0], bins=bins_mean_ro, color='grey');
        plt.hist(YY_dict[j][:,0], color='grey');
        plt.ylabel('Frequency of samples'), plt.xlabel('Mean Runoff (Mt/y)')
        plt.subplot(num_of_glaciers,2,2*j+2), plt.title('Distribution of Runoff Std', loc='left'), 
        # plt.hist(YY_dict[j][:,1], bins=bins_std_ro, color='grey');
        plt.hist(YY_dict[j][:,1], color='grey');
        plt.ylabel('Frequency of samples'), plt.xlabel('Std Runoff')
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "frequency_mean_vs_std_runoff.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

##############################################################
# Initial PAWN Sensitivity!!
##############################################################
def runoff_pawn_plot_all(params_dict, YY_dict, metric='Mean Runoff', n=5, Nboot=500, k=0):
    # Select output metric
    if metric == 'Mean Runoff':
        i = 0
    elif metric == 'Std of Runoff':
        i = 1
    else:
        raise ValueError("metric must be 'Mean Runoff' or 'Std of Runoff'")
    Y = YY_dict[k][:, i]
    X_labels = ['melt_f', 'prcp_fac', 'temp_bias']
    # TODO: When more glaciers, run this per glacier
    KS_median, KS_mean, KS_max = PAWN.pawn_indices(params_dict[0], Y, n, Nboot=Nboot)

    # Aggregate bootstrap samples
    KS_median_m, KS_median_lb, KS_median_ub = aggregate_boot(KS_median)
    KS_mean_m,   KS_mean_lb,   KS_mean_ub   = aggregate_boot(KS_mean)
    KS_max_m,    KS_max_lb,    KS_max_ub    = aggregate_boot(KS_max)
    # Plot all in one figure 
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"PAWN Sensitivity Indices\nMetric: {metric}, n={n}, Nboot={Nboot}", fontsize=14)
    # Median KS
    plt.sca(axs[0])
    pf.boxplot1(KS_median_m, S_lb=KS_median_lb, S_ub=KS_median_ub,
                X_Labels=X_labels, Y_Label='median KS')
    # Mean KS
    plt.sca(axs[1])
    pf.boxplot1(KS_mean_m, S_lb=KS_mean_lb, S_ub=KS_mean_ub,
                X_Labels=X_labels, Y_Label='mean KS')
    # Max KS
    plt.sca(axs[2])
    pf.boxplot1(KS_max_m, S_lb=KS_max_lb, S_ub=KS_max_ub,
                X_Labels=X_labels, Y_Label='max KS')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Save PNG 
    outpath = os.path.join(cfg.PATHS['working_dir'], "PAWN_plots.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"Saved sensitivity plots to {outpath}")

##############################################################
# Constraining the parameters - initial investigation
##############################################################
def plot_parameter_bounding(params_dict, successful_params_dict, area_dict, mass_balance_dict, hugonnet_dmdt, hugonnet_err_dmdt, rgi_area_km2s, rgi_dates, years_dict, num_of_glaciers, N):
    mass_balance_dict_in_period = {}
    for j in range(num_of_glaciers):
        yrs_idx = np.where((years_dict[j][0] >= 2000) & (years_dict[j][0] <= 2020))[0].tolist()

        mb_values = []
        for i in range(len(years_dict[j])):
            mbs = mass_balance_dict[j][i][yrs_idx]

            dmdt_ice = mbs.sum() / len(yrs_idx) # kg ice yr-1
            dmdt_we  = dmdt_ice * (1000.0 / cfg.PARAMS['ice_density'])
            mb_values.append(dmdt_we)

            mass_balance_dict_in_period[j] = mb_values
        
    fig, axs = plt.subplots(
        num_of_glaciers, 1,
        figsize=(8, 4*num_of_glaciers),
        squeeze=False
    )

    for j in range(num_of_glaciers):
        ax = axs[j, 0]
        upper_bound = hugonnet_dmdt[j] + hugonnet_err_dmdt[j]
        lower_bound = hugonnet_dmdt[j] - hugonnet_err_dmdt[j]
        
        # PLOT INTO THE AXES
        ax.hist(mass_balance_dict_in_period[j], bins=30, color='grey', edgecolor='white')
        # Shade region |mean| < threshold
        ax.axvspan(lower_bound, upper_bound, color='teal', alpha=0.25, label='Specific Mass Balance within Hugonnet Error Bounds')
        # vertical lines
        ax.axvline(hugonnet_dmdt[j], color='teal')
        ax.axvline(lower_bound, color='teal', linestyle='--')
        ax.axvline(upper_bound, color='teal', linestyle='--')
        # labels + title
        ax.set_xlabel("Mean Mass Balance over 2000-2020")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Glacier {j}: Distribution of Specific Mass Balance in years 2000-2020")
        ax.legend()
    
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "specific_mb_dist.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    
    area_at_rgi_yr_dict = {}
    area_before_rgi_yr_dict = {}
    area_after_rgi_yr_dict = {}
    yrs_idx = []
    for j in range(num_of_glaciers):
        yrs_idx.append(np.where(years_dict[j][0] == rgi_dates[j])[0].tolist()[0])

    for j in range(num_of_glaciers):  
        area_at_rgi_yr = []
        area_before_rgi_yr = []
        area_after_rgi_yr = [] 

        for i in range(len(area_dict[j])):
            area_at_rgi_yr.append(area_dict[j][i][yrs_idx[j]])
            area_before_rgi_yr.append(area_dict[j][i][yrs_idx[j]-1])
            area_after_rgi_yr.append(area_dict[j][i][yrs_idx[j]+1])

        area_at_rgi_yr_dict[j] = area_at_rgi_yr
        area_before_rgi_yr_dict[j] = area_before_rgi_yr
        area_after_rgi_yr_dict[j] = area_after_rgi_yr
    fig, axs = plt.subplots(
        num_of_glaciers, 1,
        figsize=(8, 4*num_of_glaciers),
        squeeze=False
    )
    for j in range(num_of_glaciers):
        ax = axs[j, 0]
        areas = [area_at_rgi_yr_dict[j][i] for i in range(len(area_at_rgi_yr_dict[j]))]
        before_areas = [area_before_rgi_yr_dict[j][i] for i in range(len(area_before_rgi_yr_dict[j]))]
        after_areas = [area_after_rgi_yr_dict[j][i] for i in range(len(area_after_rgi_yr_dict[j]))]
        print(j, len(before_areas), len(areas), len(after_areas))
        data_all = np.concatenate([areas, before_areas, after_areas])
        bins = np.linspace(np.nanmin(data_all), np.nanmax(data_all), 31)
        # plotting
        ax.hist(before_areas, bins=bins, color='grey', edgecolor='white', label="Year Before RGI: "+str(rgi_dates[j]-1))
        ax.hist(areas, bins=bins, color='blue', edgecolor='white', alpha=0.25, label="RGI Year: "+str(rgi_dates[j]))
        ax.hist(after_areas, bins=bins, color='red', edgecolor='white', alpha=0.25, label="Year After RGI: "+str(rgi_dates[j]+1))
        # labels + title
        ax.set_xlabel(r"Area (km$^2$)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Glacier {j}: Distribution of Areas at the RGI year, and adjacent years.")
        ax.legend()
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "area_dist_at_rgi.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    
    # Now constraining the parameters 
    new_lower_bounds_list = []
    new_upper_bounds_list = []
    good_X_list = []
    for j in range(num_of_glaciers):
        new_lower_bounds, new_upper_bounds, good_X = parameter_bounding(successful_params_dict[j], 
                                                                area_dict[j], 
                                                                mass_balance_dict_in_period[j],
                                                                hugonnet=hugonnet_dmdt[j],
                                                                hugonnet_error=hugonnet_err_dmdt[j],
                                                                obs_area=rgi_area_km2s[j],
                                                                year_idx=yrs_idx[j],
                                                                area_percentile=10,
                                                                area_bounding_flag=True,
                                                                hugonnet_bounding_flag=True)
        new_lower_bounds_list.append(new_lower_bounds)
        new_upper_bounds_list.append(new_upper_bounds)
        good_X_list.append(good_X)
        print("The new upper bounds for glacier %d are: %s" % (j, new_upper_bounds))
        print("The new lower bounds for glacier %d are: %s" % (j, new_lower_bounds))
    X_labels = ['melt_f', 'prcp_fac', 'temp_bias']
    M = len(X_labels)
    fig, axs = plt.subplots(
        num_of_glaciers, M,
        figsize=(15, 4*num_of_glaciers),
        squeeze=False
    )
    for j in range(num_of_glaciers):
        # loop over parameters and plot scatter
        for i in range(M):
            ax = axs[j, i]
            ax.scatter(
                params_dict[j][:, i],
                params_dict[j][:, (i+1) % M],
                s=20,
                edgecolors='none',
                color='grey',
                alpha=0.5
            )
            ax.scatter(
                good_X_list[j][:, i],
                good_X_list[j][:, (i+1) % M],
                s=20,
                edgecolors='none',
                color='red'
            )
            ax.set_xlabel(X_labels[i])
            ax.set_ylabel(X_labels[(i+1) % M])
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "reduced_bounds.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    return good_X_list

def plot_parameter_bounding_3d(params_dict, good_X_list, num_of_glaciers):
# -----------------------------
# 3. 3D scatter plots
# -----------------------------
    fig3d = plt.figure(figsize=(6, 4 * num_of_glaciers))
    axs3d = []
    for j in range(num_of_glaciers):
        ax3d = fig3d.add_subplot(
            num_of_glaciers, 1, j + 1, projection='3d'
        )
        axs3d.append(ax3d)
        ax3d.scatter(
            params_dict[j][:, 0],
            params_dict[j][:, 1],
            params_dict[j][:, 2],
            s=10,
            color='grey',
            alpha=0.3
        )
        ax3d.scatter(
            good_X_list[j][:, 0],
            good_X_list[j][:, 1],
            good_X_list[j][:, 2],
            s=25,
            color='red'
        )
        ax3d.set_xlabel('melt_f')
        ax3d.set_ylabel('prcp_fac')
        ax3d.set_zlabel('temp_bias')
        ax3d.set_title(f'Glacier {j}')
    # -----------------------------
    # 4. Save rotated views (cluster-safe)
    # -----------------------------
    views = [(20, 30), (20, 120), (60, 30)]
    for elev, azim in views:
        for ax in axs3d:
            ax.view_init(elev=elev, azim=azim)
        plt.savefig(
            os.path.join(
                cfg.PATHS['working_dir'],
                f"reduced_bounds_3D_e{elev}_a{azim}.png"
            ),
            dpi=200,
            bbox_inches='tight'
        )
    plt.close(fig3d)

if __name__ == "__main__":
        main()
