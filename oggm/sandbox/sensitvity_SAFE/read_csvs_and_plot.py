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
                        help="Parameters CSV path")

    parser.add_argument("--x_max", type=float, nargs=6, required=True)
    
    parser.add_argument("--x_min", type=float, nargs=6, required=True)

    parser.add_argument("--area_uncertainty", type=float, nargs='+', required=True)

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
    rgi_id = args.rgi_ids
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup'
    gdir = workflow.init_glacier_directories(rgi_id, from_prepro_level=4, prepro_border=160, prepro_base_url=base_url)[0]
    
    N = args.N
    rgi_date = gdir.rgi_date
    rgi_area_km2 = gdir.rgi_area_km2

    # And match the Hugonnet
    geo_df = utils.get_geodetic_mb_dataframe()

    mask = geo_df['period'].eq('2000-01-01_2020-01-01')
    selected_gdirs_geo_df = geo_df.loc[geo_df.index.isin([str(rgi_id[0])]) & mask]

    # Convert the Hugonnet Observations to dmdt
    hugonnet_dmdt = selected_gdirs_geo_df['dmdtda'].values[0] * rgi_area_km2
    hugonnet_err_dmdt = selected_gdirs_geo_df['err_dmdtda'].values[0] * rgi_area_km2

    area_uncertainty = args.area_uncertainty[0]
    
    ##############################################################
    # Now call all of our functions!
    ##############################################################
    mass_balance_dict, years_dict, runoff_dict, area_dict, volume_dict, params_samples, params_valid = read_csvs(args, N)

    plot_mass_balance_timeseries(mass_balance_dict, years_dict, rgi_id, N)
    plot_area_timeseries(area_dict, years_dict, rgi_id, N, gdir, area_uncertainty)
    plot_volume_timeseries(volume_dict, years_dict, rgi_id, N)
    plot_runoff_timeseries(runoff_dict, years_dict, rgi_id, N)
    plot_input_hists(params_samples, args)
    plot_input_distributions(N, mass_balance_dict, rgi_id)

    YY_dict = hydro_output_calculator(runoff_dict)
    plot_mean_vs_std_runoff(YY_dict)
    plot_runoff_mean_and_std_distributions(YY_dict)
    runoff_pawn_plot_all(params_valid, YY_dict, metric='Mean Runoff', n=5, Nboot=500, k=0)

    plot_parameter_bounding(args, params_valid, params_samples, area_dict, area_uncertainty, mass_balance_dict, hugonnet_dmdt, hugonnet_err_dmdt, rgi_area_km2, rgi_date, years_dict, rgi_id)

    # param_names = ['melt_f', 'prcp_fac', 'temp_bias', 'glen_a_fac', 'phase_shift', 'temp_melt']
    # prior_bounds = [(1.5, 17), (0.1, 10), (-15, 15), (1.0, 1.5), (-3, 3), (-1.5, 0.5)]

    # fig, ax = plot_prior_posterior_ranges(
    #     csv_path=cfg.PATHS['working_dir'] + '/' + 'reduced_bounds.csv',
    #     param_names=param_names,
    #     prior_bounds=prior_bounds,
    #     glacier_id=args.rgi_ids[0],
    #     outpath=os.path.join(cfg.PATHS['working_dir'], "prior_posterior_ranges.png")
    # )

##############################################################
# Read CSVs
##############################################################
def read_csvs(args, N):

    # Outputs
    mass_balance_samples = []
    years_samples = []
    runoff_samples = []
    area_samples = []
    volume_samples = []

    # Parameters
    params_all = [] # All parameter samples
    params_valid = [] # Successful parameter samples

    for i in range(N):

        param_path = cfg.PATHS['working_dir'] + '/' + str(i) + args.params_csv_path
        output_path = cfg.PATHS['working_dir'] + '/' + str(i) + args.output_csv_path

        # Read params
        try:
            
            mb_param_df = pd.read_csv(param_path)
            row = mb_param_df[["melt_f", "prcp_fac", "temp_bias", "glen_a_fac", "phase_shift", "temp_melt"]].values[0]
            params_all.append(row)
        except FileNotFoundError:
            print(f"No parameter file for index {i}")
            continue

        # Read outputs if they exist
        try:
            df = pd.read_csv(output_path)

            if np.all(np.abs(df['mass_balance'].values[1:]) < 1e-12):
                continue

            area_samples.append(df['area_km2'].values[1:])
            runoff_samples.append(df['runoff_Mt'].values[1:])
            mass_balance_samples.append(df['mass_balance'].values[1:])
            volume_samples.append(df['volume_km3'].values[1:])
            years_samples.append(df['years'].values[1:])

            # Parameters for successful simulations
            params_valid.append(row)

        except FileNotFoundError:
            print(output_path)
            print(f"Output missing for index {i}")

    # convert to arrays
    params_all = np.array(params_all)
    params_valid = np.array(params_valid)

    return (mass_balance_samples, years_samples, runoff_samples,
            area_samples, volume_samples,
            params_all, params_valid)

##############################################################
# Plotting Simulation Outputs
##############################################################
def plot_mass_balance_timeseries(mass_balance_samples, years_samples, rgi_id, N):
    # compile csvs to plot timeseries and plot to view the mass balance time series for each of the samples, to see how they are looking and check that they make sense before we calculate the sensitivity indices
    plt.figure(figsize=(13,4))

    for i in range(len(years_samples)):
        plt.subplot(1,1,1)
        plt.plot(years_samples[i], mass_balance_samples[i], label='sim', color='k', linewidth=0.5, alpha=0.1)
    plt.title('Mass balance time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_id))
    plt.xlabel('Year'), plt.ylabel('Mass Balance kg m$^-2$')
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "mass_balance.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

def plot_area_timeseries(area_samples, years_samples, rgi_id, N, gdir, area_unc_percent):
    plt.figure(figsize=(13,4))

    for i in range(len(years_samples)):
        plt.subplot(1,1,1)
        plt.plot(years_samples[i], area_samples[i], label='sim', color='k', linewidth=0.5, alpha=0.1)

    unc = area_unc_percent / 100.0

    lower = gdir.rgi_area_km2 - (gdir.rgi_area_km2*unc)
    upper = gdir.rgi_area_km2 + (gdir.rgi_area_km2*unc)

    plt.fill_between(
        [years_samples[0][0], years_samples[0][-1]],  # full time range
        lower,
        upper,
        color='red',
        alpha=0.5,
        label=f'Observed ±{area_unc_percent}%'
        )

    plt.scatter(
        gdir.rgi_date,
        gdir.rgi_area_km2,
        color='red',
        s=40,
        zorder=999
    )

    plt.axhspan(
        lower,
        upper,
        color='red',
        alpha=0.9,
        xmin=gdir.rgi_date-1,
	xmax=gdir.rgi_date+1
    )

    plt.plot(
        [years_samples[0][0], years_samples[0][-1]],
        [lower, lower],
        color='red',
        linewidth=2,
        zorder=5
    )

    plt.plot(
        [years_samples[0][0], years_samples[0][-1]],
        [upper, upper],
        color='red',
        linewidth=2,
        zorder=5
    )

   
    plt.title('Area time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_id))
    plt.ylabel('Glacier area (km$^2$)')
    plt.xlabel('Year')

    plt.xlim(years_samples[0][0], years_samples[0][-1])

    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "area_post.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.figure(figsize=(13,4))

def plot_volume_timeseries(volume_samples, years_samples, rgi_id, N):
    plt.figure(figsize=(13,4))
    
    for i in range(len(years_samples)):
        plt.subplot(1,1,1)
        plt.plot(years_samples[i], volume_samples[i], label='sim', color='k', linewidth=0.5, alpha=0.1)
    plt.title('Volume time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_id))
    plt.ylabel('Glacier volume (km$^3$)')
    plt.xlabel('Year')
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "volume.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.figure(figsize=(13,4))

def plot_runoff_timeseries(runoff_samples, years_samples, rgi_id, N):
    plt.figure(figsize=(13,4))

    for i in range(len(years_samples)):
        plt.subplot(1,1,1)
        plt.plot(years_samples[i], runoff_samples[i], label='sim', color='k', linewidth=0.5, alpha=0.1)

    plt.title('Runoff time series for each parameter sample, N = %d for RGI-ID = %s' % (N, rgi_id))

    plt.ylabel('Runoff (Mt/yr)')
    plt.xlabel('Year')
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "runoff_post.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

##############################################################
# Plotting distribution of Inputs
##############################################################
def plot_input_hists(params_samples, args):

    x_max = args.x_max
    x_min = args.x_min

    melt_low, melt_high = x_min[0], x_max[0]
    precip_low, precip_high = x_min[1], x_max[1]
    tbias_low, tbias_high = x_min[2], x_max[2]
    glen_a_low, glen_a_high = x_min[3], x_max[3]
    phase_shift_low, phase_shift_high = x_min[4], x_max[4]
    temp_melt_low, temp_melt_high = x_min[5], x_max[5]

    nbins = 50
    bins_melt   = np.linspace(melt_low,   melt_high,   nbins+1)
    bins_precip = np.linspace(precip_low, precip_high, nbins+1)
    bins_tbias  = np.linspace(tbias_low,  tbias_high,  nbins+1)
    bins_glen_a = np.linspace(glen_a_low, glen_a_high, nbins+1)
    bins_phase_shift = np.linspace(phase_shift_low, phase_shift_high, nbins+1)
    bins_temp_melt = np.linspace(temp_melt_low, temp_melt_high, nbins+1)
    
    plt.figure(figsize=(13, 4))
    plt.subplot(1,6,1)
    plt.title('Distribution of melt factor', loc='left')
    params_samples = np.array(params_samples)
    plt.hist(params_samples[:,0], bins=bins_melt, range=(melt_low, melt_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Melt Factor')
    plt.subplot(1,6,2)
    plt.title('Distribution of precipitation factor', loc='left')
    plt.hist(params_samples[:,1], bins=bins_precip, range=(precip_low, precip_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Precipitation Factor')
    plt.subplot(1,6,3)
    plt.title('Distribution of temperature bias', loc='left')
    plt.hist(params_samples[:,2], bins=bins_tbias, range=(tbias_low, tbias_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Temperature Bias')
    plt.subplot(1,6,4)
    plt.title('Distribution of Glen A multiplier', loc='left')
    plt.hist(params_samples[:,3], bins=bins_glen_a, range=(glen_a_low, glen_a_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Glen A Multiplier')
    plt.tight_layout()
    plt.subplot(1,6,5)
    plt.title('Distribution of Phase Shift', loc='left')
    plt.hist(params_samples[:,4], bins=bins_phase_shift, range=(phase_shift_low, phase_shift_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Phase Shift')
    plt.tight_layout()
    plt.subplot(1,6,6)
    plt.title('Distribution of Temp Melt', loc='left')
    plt.hist(params_samples[:,5], bins=bins_temp_melt, range=(temp_melt_low, temp_melt_high),
             color='grey', edgecolor='white')
    plt.ylabel('Frequency of samples'); plt.xlabel('Temp Melt')
    plt.tight_layout()

    outpath = os.path.join(cfg.PATHS['working_dir'], "parameter_distribution.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.tight_layout()
    
    outpath = os.path.join(cfg.PATHS['working_dir'], "parameter_distribution.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')


##############################################################
# Plotting Mass Balance Means and Standard Deviations
##############################################################
def plot_input_distributions(N, mass_balance_samples, rgi_id):
    mean_mass_balances_list = []
    std_mass_balances_list  = []
    
    for i in range(len(mass_balance_samples)): # number of samples
        mb_series = mass_balance_samples[i]
        mean_mass_balances_list.append(mb_series.mean())
        std_mass_balances_list.append(mb_series.std())

    plt.figure(figsize=[13,4])

    min_means, max_means = np.nanmin(mean_mass_balances_list), np.nanmax(mean_mass_balances_list)
    min_stds, max_stds = np.nanmin(std_mass_balances_list), np.nanmax(std_mass_balances_list)
    nbins = 100
    bins_means = np.linspace(min_means, max_means, nbins+1)
    bins_stds = np.linspace(min_stds, max_stds, nbins+1)
    plt.subplot(1,2,1), 
    plt.title('Distribution of Mean Mass Balance for RGI_ID = %s ' % rgi_id, loc='left')
    plt.hist(mean_mass_balances_list, range=(min_means, max_means), color='grey');
    plt.ylabel('Frequency of samples'), plt.xlabel('Mean')
    # plt.hist(mean_mass_balances_dict[j], bins='auto', range=(min_means, max_means), color='grey');
    # plt.ylabel('Frequency of samples'), plt.xlabel('Mean')
    # plt.subplot(num_of_glaciers,2,2*j+2), plt.title('Distribution of Std of Mass Balance', loc='left'), plt.hist(std_mass_balances_dict[j], bins='auto', 
    #                                                                                          range=(min_stds, max_stds), color='grey');
    plt.subplot(1,2,2), plt.title('Distribution of Std of Mass Balance', loc='left'), plt.hist(std_mass_balances_list, bins='auto', color='grey');
    plt.ylabel('Frequency of samples'), plt.xlabel('Std')
    plt.tight_layout()

    outpath = os.path.join(cfg.PATHS['working_dir'], "output_distributions.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

def hydro_output_calculator(runoff_dict):
    YY_list = hydro_output_metric_calculator(runoff_dict)
    return YY_list

##############################################################
# Plot of Mean vs Std Runoff values
##############################################################
def plot_mean_vs_std_runoff(YY_list):
    plt.figure(figsize=[13,4])

    plt.subplot(1,3,2), plt.scatter(YY_list[:,0], YY_list[:,1], s=5), plt.title('Mean runoff vs Std of runoff'), plt.xlabel('Mean runoff Mt/yr'), plt.ylabel('Std of runoff Mt/yr'),
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "mean_vs_std_runoff.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
##############################################################
# Histogram plot of Runoff Mean and Std
##############################################################
def plot_runoff_mean_and_std_distributions(YY_list):

    plt.figure(figsize=[13,4])

    min_mean_ro, max_mean_ro = np.min(YY_list[:,0]), np.max(YY_list[:,0])
    min_std_ro, max_std_ro = np.min(YY_list[:,1]), np.max(YY_list[:,1])
    nbins = 50
    bins_mean_ro = np.linspace(min_mean_ro, max_mean_ro, nbins+1)
    bins_std_ro = np.linspace(min_std_ro, max_std_ro, nbins+1)
    plt.subplot(1,2,1), plt.title('Distribution of Runoff Mean', loc='left'), 
    # plt.hist(YY_dict[j][:,0], bins=bins_mean_ro, color='grey');
    plt.hist(YY_list[:,0], color='grey');
    plt.ylabel('Frequency of samples'), plt.xlabel('Mean Runoff (Mt/y)')
    plt.subplot(1,2,2), plt.title('Distribution of Runoff Std', loc='left'), 
    # plt.hist(YY_dict[j][:,1], bins=bins_std_ro, color='grey');
    plt.hist(YY_list[:,1], color='grey');
    plt.ylabel('Frequency of samples'), plt.xlabel('Std Runoff')

    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "frequency_mean_vs_std_runoff.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

##############################################################
# Initial PAWN Sensitivity!!
##############################################################
def runoff_pawn_plot_all(params_valid, YY_list, metric='Mean Runoff', n=5, Nboot=500, k=0):
    # Select output metric
    if metric == 'Mean Runoff':
        i = 0
    elif metric == 'Std of Runoff':
        i = 1
    else:
        raise ValueError("metric must be 'Mean Runoff' or 'Std of Runoff'")
    Y = YY_list[:, i]
    X_labels = ['melt_f', 'prcp_fac', 'temp_bias', 'glen_a_fac', 'phase_shift', 'temp_melt']
    params_samples = np.array(params_valid)

    KS_median, KS_mean, KS_max = PAWN.pawn_indices(params_valid, Y, n, Nboot=Nboot)

    # Aggregate bootstrap samples
    KS_median_m, KS_median_lb, KS_median_ub = aggregate_boot(KS_median)
    KS_mean_m,   KS_mean_lb,   KS_mean_ub   = aggregate_boot(KS_mean)
    KS_max_m,    KS_max_lb,    KS_max_ub    = aggregate_boot(KS_max)
    # Plot all in one figure 
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
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
def plot_parameter_bounding(args, params_valid,params_samples, area_samples, area_uncertainty, mass_balance_samples, hugonnet_dmdt, hugonnet_err_dmdt, rgi_area_km2, rgi_date, years_samples, rgi_id):

    yrs_idx = np.where((years_samples[0] >= 2000) & (years_samples[0] <= 2020))[0].tolist()

    mean_mb_values = []
    mean_mb_values_scaled = []
    
    for i in range(len(years_samples)):
        mbs = mass_balance_samples[i][yrs_idx]

        mean_mb_values.append(np.mean(mbs))
        mean_mb_values_scaled.append((np.mean(mbs))/ rgi_area_km2)
        
    fig, axs = plt.subplots(
        1, 1,
        figsize=(8, 4),
        squeeze=False
    )

    ax = axs[0, 0]
    upper_bound = hugonnet_dmdt + hugonnet_err_dmdt
    lower_bound = hugonnet_dmdt - hugonnet_err_dmdt
    
    # Plotting
    ax.hist(mean_mb_values_scaled, bins=30, color='grey', edgecolor='white')

    # Shade region |mean| < threshold
    ax.axvspan(lower_bound/rgi_area_km2, upper_bound/rgi_area_km2, color='teal', alpha=0.15, label='Hugonnet Error Bounds')
    # vertical lines
    ax.axvline(hugonnet_dmdt/rgi_area_km2, color='teal')
    ax.axvline(lower_bound/rgi_area_km2, color='teal', linestyle='--')
    ax.axvline(upper_bound/rgi_area_km2, color='teal', linestyle='--')
    # labels + title
    ax.set_xlabel("Mean Glacier Mass Balance over 2000-2020, normalised by glacier area (kg km⁻² yr⁻¹)")
    ax.set_ylabel("Frequency")
    ax.set_title('Distribution of Mean Glacier Mass Change (2000–2020) for RGI_ID = %s' % rgi_id)
    ax.legend()
    
    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "specific_mb_dist.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

    yrs_area_idx = np.where(years_samples[0] == rgi_date)[0][0]

    areas = []
    before_areas = []
    after_areas = [] 
    for i in range(len(area_samples)):
        areas.append(area_samples[i][yrs_area_idx])
        before_areas.append(area_samples[i][yrs_area_idx-1])
        after_areas.append(area_samples[i][yrs_area_idx+1])

    fig, axs = plt.subplots(
        1, 1,
        figsize=(8, 4),
        squeeze=False
    )

    ax = axs[0, 0]
    data_all = np.concatenate([areas, before_areas, after_areas])
    bins = np.linspace(np.nanmin(data_all), np.nanmax(data_all), 31)
    
    # plotting
    ax.hist(before_areas, bins=bins, color='grey', edgecolor='white', label="Year Before RGI: "+str(rgi_date-1))
    ax.hist(areas, bins=bins, color='blue', edgecolor='white', alpha=0.25, label="RGI Year: "+str(rgi_date))
    ax.hist(after_areas, bins=bins, color='red', edgecolor='white', alpha=0.25, label="Year After RGI: "+str(rgi_date+1))
    
    # labels + title
    ax.set_xlabel(r"Area (km$^2$)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Glacier: Distribution of Areas at the RGI year, and adjacent years.")
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "area_dist_at_rgi.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    
    # Now constraining the parameters 
    new_lower_bounds_list = []
    new_upper_bounds_list = []
    good_X_list = []

    new_lower_bounds, new_upper_bounds, good_X = parameter_bounding(params_valid, 
                                                            area_samples, 
                                                            mean_mb_values,
                                                            hugonnet=hugonnet_dmdt,
                                                            hugonnet_error=hugonnet_err_dmdt,
                                                            obs_area=rgi_area_km2,
                                                            year_idx=yrs_area_idx,
                                                            area_percentile=area_uncertainty,
                                                            area_bounding_flag=True,
                                                            hugonnet_bounding_flag=True)
    new_lower_bounds_list.append(new_lower_bounds)
    new_upper_bounds_list.append(new_upper_bounds)
    good_X_list.append(good_X)

    if new_lower_bounds is None or new_upper_bounds is None:
        print("No new bounds could be calculated for glacier, likely because no samples satisfied the bounding criteria.")
    else:
        print("The new upper bounds for glacier are: %s" % (new_upper_bounds))
        print("The new lower bounds for glacier are: %s" % (new_lower_bounds))

    from itertools import combinations

    X_labels = ['melt_f', 'prcp_fac', 'temp_bias', 'glen_a_fac', 'phase_shift', 'temp_melt']
    M = len(X_labels)

    # all unique pairs (i, j) with i < j
    pairs = list(combinations(range(M), 2))
    n_plots = len(pairs)

    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    params_samples = np.asarray(params_samples)
    good_X_list = np.asarray(good_X_list)
    good_X_list = np.squeeze(good_X_list)

    # Ensure it's always 2D: (n_samples, n_params)
    if good_X_list.ndim == 1:
        good_X_list = good_X_list.reshape(1, -1)

    # loop over every parameter pair
    for k, (i, j) in enumerate(pairs):
        row, col = divmod(k, n_cols)
        ax = axs[row, col]

        ax.scatter(
            params_samples[:, i],
            params_samples[:, j],
            s=20,
            edgecolors='none',
            color='grey',
            alpha=0.5
        )

        if good_X_list.size > 0:
            ax.scatter(
                good_X_list[:, i],
                good_X_list[:, j],
                s=20,
                edgecolors='none',
                color='red'
            )

        ax.set_xlabel(X_labels[i])
        ax.set_ylabel(X_labels[j])

    # hide any unused subplot axes (blank spaces)
    for k in range(n_plots, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axs[row, col].axis('off')

    plt.tight_layout()
    outpath = os.path.join(cfg.PATHS['working_dir'], "reduced_bounds.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight')

    if good_X_list.size > 0:
        csv_outpath = os.path.join(cfg.PATHS['working_dir'], "reduced_bounds.csv")

        # Save the new bounds to a CSV file
        with open(csv_outpath, 'w') as f:
            f.write("xmin1,xmin2,xmin3,xmin4,xmin5,xmin6,xmax1,xmax2,xmax3,xmax4,xmax5,xmax6\n")

            if new_lower_bounds_list is None or new_upper_bounds_list is None:
                xmin = args.x_min
                xmax = args.x_max
            else:
                xmin = new_lower_bounds_list[0]
                xmax = new_upper_bounds_list[0]

            f.write(f"{xmin[0]},{xmin[1]},{xmin[2]},{xmin[3]},{xmin[4]},{xmin[5]},{xmax[0]},{xmax[1]},{xmax[2]},{xmax[3]},{xmax[4]},{xmax[5]}\n")

    return good_X_list

# def plot_parameter_bounding_3d(params_dict, good_X_list, num_of_glaciers):
# # -----------------------------
# # 3D scatter plots
# # -----------------------------
#     fig3d = plt.figure(figsize=(6, 4 * num_of_glaciers))
#     axs3d = []
#     for j in range(num_of_glaciers):
#         ax3d = fig3d.add_subplot(
#             num_of_glaciers, 1, j + 1, projection='3d'
#         )
#         axs3d.append(ax3d)
#         ax3d.scatter(
#             params_dict[j][:, 0],
#             params_dict[j][:, 1],
#             params_dict[j][:, 2],
#             s=10,
#             color='grey',
#             alpha=0.3
#         )
#         ax3d.scatter(
#             good_X_list[j][:, 0],
#             good_X_list[j][:, 1],
#             good_X_list[j][:, 2],
#             s=25,
#             color='red'
#         )
#         ax3d.set_xlabel('melt_f')
#         ax3d.set_ylabel('prcp_fac')
#         ax3d.set_zlabel('temp_bias')
#         ax3d.set_title(f'Glacier {j}')
#     # -----------------------------
#     # Save rotated views
#     # -----------------------------
#     views = [(20, 30), (20, 120), (60, 30)]
#     for elev, azim in views:
#         for ax in axs3d:
#             ax.view_init(elev=elev, azim=azim)
#         plt.savefig(
#             os.path.join(
#                 cfg.PATHS['working_dir'],
#                 f"reduced_bounds_3D_e{elev}_a{azim}.png"
#             ),
#             dpi=200,
#             bbox_inches='tight'
#         )
#     plt.close(fig3d)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def plot_prior_posterior_ranges(
    csv_path,
    param_names,
    prior_bounds,
    glacier_id=None,
    bar_height=6,
    figsize=(7, 4),
    title="Prior vs Posterior Parameter Range Comparison",
    outpath=None,
):
    """
    Reads a single glacier's posterior (reduced) parameter bounds from a CSV
    in the format:
        xmin1, xmin2, ..., xminN, xmax1, xmax2, ..., xmaxN
    (one row), and plots horizontal prior vs posterior range bars for that
    glacier.

    Parameters
    ----------
    csv_path : str
        Path to the glacier's reduced-bounds CSV.
    param_names : list of str
        Parameter labels, in the SAME ORDER as the xmin/xmax column indices
        (e.g. param_names[0] corresponds to xmin1/xmax1), and in the order
        they should appear top-to-bottom on the y-axis.
    prior_bounds : list of (low, high) tuples
        The prior (original sampling) range for each parameter, in the same
        order as param_names.
    glacier_id : str, optional
        Label used as the plot title. Defaults to the CSV's filename
        (without extension) if not given.
    bar_height : float
        Line width (in points) used to draw each range as a thick bar.
    figsize : tuple
        Figure size.
    title : str
        Overall figure title (shown above the glacier_id subtitle).
    outpath : str, optional
        If given, saves the figure to this path.

    Returns
    -------
    fig, ax
    """

    n_params = len(param_names)
    xmin_cols = [f'xmin{i+1}' for i in range(n_params)]
    xmax_cols = [f'xmax{i+1}' for i in range(n_params)]

    df = pd.read_csv(csv_path)
    missing = [c for c in xmin_cols + xmax_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}")

    row = df.iloc[0]  # single-row CSV
    posterior_ranges = [(row[xmin_cols[j]], row[xmax_cols[j]]) for j in range(n_params)]

    if glacier_id is None:
        import os
        glacier_id = os.path.splitext(os.path.basename(csv_path))[0]

    fig, ax = plt.subplots(figsize=figsize)

    # y-positions: first param_name at the TOP
    y_pos = np.arange(n_params)[::-1]

    for y, (p_low, p_high), (post_low, post_high) in zip(
        y_pos, prior_bounds, posterior_ranges
    ):
        # Prior range (grey, drawn first so posterior sits on top)
        ax.hlines(y, p_low, p_high, color='lightgrey',
                  linewidth=bar_height, zorder=1)
        # Posterior range (blue)
        ax.hlines(y, post_low, post_high, color='blue',
                  linewidth=bar_height, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel("Parameter Range")
    ax.set_ylabel("Parameters")
    ax.set_title(glacier_id)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_ylim(-1, n_params)

    legend_handles = [
        mlines.Line2D([], [], color='lightgrey', linewidth=bar_height, label='Prior'),
        mlines.Line2D([], [], color='blue', linewidth=bar_height, label='Posterior'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if outpath is not None:
        plt.savefig(outpath, dpi=200, bbox_inches='tight')

    return fig, ax

if __name__ == "__main__":
        main()
