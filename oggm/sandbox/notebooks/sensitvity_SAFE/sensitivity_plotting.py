import pandas as pd
from oggm import cfg
import numpy as np
import matplotlib.pyplot as plt
import os
import oggm
from oggm import cfg, workflow, utils
from oggm.core.sensitivity import hydro_output_metric_calculator

from oggm.core.sensitivity import parameter_bounding
from oggm.sandbox.notebooks.sensitvity_SAFE.read_csvs_and_plot import read_csvs


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
    mass_balance_dict, years_dict, runoff_dict, area_dict, volume_dict, params_dict = read_csvs(args, num_of_glaciers, gdirs, N)

    N = N-1
    execute_parameter_bounding(params_dict, area_dict, mass_balance_dict, hugonnet_dmdt, hugonnet_err_dmdt, rgi_area_km2s, rgi_dates, years_dict, num_of_glaciers, N)


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
    for j in range(num_of_glaciers):
        mass_balance_samples = []
        years_samples = []
        runoff_samples = []
        area_samples = []
        volume_samples = []
        params = []
        statuses = []
        for i in range(N):
            try:
                path = cfg.PATHS['working_dir'] + '/' + str(i) + args.output_csv_path
                df = pd.read_csv(path)
                # Append arrays
                area_samples.append(df['area_km2'].values[1:])
                runoff_samples.append(df['runoff_Mt'].values[1:])
                mass_balance_samples.append(df['mass_balance'].values[1:])
                volume_samples.append(df['volume_km3'].values[1:])
                years_samples.append(df['years'].values[1:])
                # Check for failure - if the area is zero for all years, this likely means the glacier has disappeared and the simulation has failed, so we can flag this in the statuses list and ignore these samples in the sensitivity analysis (or we could also choose to include them and see how they affect the sensitivity indices, but here we are just flagging them for now)
                if np.array(area_samples).max() == 0:
                    statuses.append("failed")
                else:
                    statuses.append("ok")
                mb_param_df = pd.read_csv(
                    cfg.PATHS['working_dir'] + '/' + str(i) + args.params_csv_path
                )
                params.append(mb_param_df['params'].values)
            except FileNotFoundError:
                print(f"No simulation for index {i}")
                statuses.append("missing")
        mass_balance_dict[j] = mass_balance_samples
        years_dict[j] = years_samples
        runoff_dict[j] = runoff_samples
        area_dict[j] = area_samples
        volume_dict[j] = volume_samples
        params_dict[j] = np.vstack(params)
        print("done glacier: ", gdirs[j].rgi_id)
    return mass_balance_dict, years_dict, runoff_dict, area_dict, volume_dict, params_dict

##############################################################
# Constraining the parameters - initial investigation
##############################################################
def execute_parameter_bounding(params_dict, area_dict, mass_balance_dict, hugonnet_dmdt, hugonnet_err_dmdt, rgi_area_km2s, rgi_dates, years_dict, num_of_glaciers, N):
    mass_balance_dict_in_period = {}
    for j in range(num_of_glaciers):
        yrs_idx = np.where((years_dict[j][0] >= 2000) & (years_dict[j][0] <= 2019))[0].tolist()
        mb_values = []
        for i in range(N):
            mbs = mass_balance_dict[j][i][yrs_idx]
            
            dmdt_ice = mbs.sum() / len(yrs_idx) # kg ice yr-1
            dmdt_we  = dmdt_ice * (1000.0 / cfg.PARAMS['ice_density'])
            mb_values.append(dmdt_we)

            mass_balance_dict_in_period[j] = mb_values
        
    # Now constraining the parameters 
    new_lower_bounds_dict = {}
    new_upper_bounds_dict = {}
    good_X_dict = {}

    for j in range(num_of_glaciers):
        new_lower_bounds, new_upper_bounds, good_X = parameter_bounding(params_dict[j], 
                                                                area_dict[j], 
                                                                mass_balance_dict_in_period[j],
                                                                hugonnet=hugonnet_dmdt[j],
                                                                hugonnet_error=hugonnet_err_dmdt[j],
                                                                obs_area=rgi_area_km2s[j],
                                                                year_idx=yrs_idx[j],
                                                                area_percentile=10,
                                                                area_bounding_flag=True,
                                                                hugonnet_bounding_flag=True)
        new_lower_bounds_dict[j] = new_lower_bounds
        new_upper_bounds_dict[j] = new_upper_bounds
        good_X_dict[j] = good_X

        print("The new upper bounds for glacier %d are: %s" % (j, new_upper_bounds))
        print("The new lower bounds for glacier %d are: %s" % (j, new_lower_bounds))
    
    return new_lower_bounds_dict, new_upper_bounds_dict, good_X_dict

def bounds_comparison(new_lower_bounds_dict, new_upper_bounds_dict, original_x_min, original_x_max, num_of_glaciers):
    lower_bounds_dict = {}
    upper_bounds_dict = {}
    for j in range(num_of_glaciers):
        if new_lower_bounds_dict[j] is not None and new_upper_bounds_dict[j] is not None:
            if new_lower_bounds_dict[j] > original_x_min or new_upper_bounds_dict[j] < original_x_max:
                lower_bounds_dict[j] = new_lower_bounds_dict[j]
                upper_bounds_dict[j] = new_upper_bounds_dict[j]
            else:
                lower_bounds_dict[j] = original_x_min
                upper_bounds_dict[j] = original_x_max
        else:
            lower_bounds_dict[j] = original_x_min
            upper_bounds_dict[j] = original_x_max

        print("Glacier %d:" % j)
        print("Original bounds: ", original_x_min, original_x_max)
        print("New lower bounds: ", new_lower_bounds_dict[j])
        print("New upper bounds: ", new_upper_bounds_dict[j])

        return lower_bounds_dict, upper_bounds_dict


if __name__ == "__main__":
        main()
