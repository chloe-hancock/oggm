import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as st
import pandas as pd

from oggm import cfg, workflow, utils
from oggm.core import flowline, sensitivity
from oggm import tasks
from oggm.core.massbalance import MultipleFlowlineMassBalance

import safepython.PAWN as PAWN # Module to calculate PAWN sensitivity indices
from safepython.sampling import AAT_sampling # Functions to perform the input sampling
from safepython.util import aggregate_boot # Functions to perform bootstrapping
import safepython.plot_functions as pf
import multiprocessing as mp
import os
import time

from oggm.core.sensitivity import hydro_output_metric_calculator, run_with_runoff_for_sa, runoff_execution, colour_plotting_timeseries,mean_diff, parameter_bounding, spinup_area_volume

def main():

        cfg.initialize(logging_level='CRITICAL')
        cfg.PATHS['working_dir'] = "~/OGGM_repo/oggm/oggm/sandbox/notebooks/sensitvity_SAFE/glacier_outs"
        cfg.PARAMS['border'] = 10
        cfg.PARAMS['store_model_geometry'] = True
        cfg.PARAMS['min_ice_thick_for_length'] = 1  # a glacier is when ice thicker than 1m

        rgi_ids = ['RGI60-14.00063', 'RGI60-11.00897']

        cfg.PARAMS['use_multiprocessing'] = True  # To speed up sensitivity analysis runs
        cfg.PARAMS['mp_processes'] = 32

        mp.set_start_method("spawn", force=True)

        # We pick the elevation-bands glaciers because they run a bit faster - but they create more step changes in the area outputs
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup'
        gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=4, prepro_border=160, prepro_base_url=base_url)

        # Get the Hugonnet mass balance and set up dataframe
        geo_df = utils.get_geodetic_mb_dataframe()
        geo_df.loc[rgi_ids]

        # Hydrological model workflow steps before running with hydro
        cfg.PARAMS['evolution_model'] = 'FluxBased'
        cfg.PARAMS['store_model_geometry'] = True
        cfg.PARAMS['error_when_glacier_reaches_boundaries'] = False

        num_of_glaciers = len(gdirs)

        rgi_dates = []
        rgi_area_km2s = []

        for gdir in gdirs:
                rgi_dates.append(gdir.rgi_date)
                rgi_area_km2s.append(gdir.rgi_area_km2)

                # And match the Hugonnet
        geo_df = utils.get_geodetic_mb_dataframe()

        mask = geo_df['period'].eq('2000-01-01_2020-01-01')
        selected_gdirs_geo_df = geo_df.loc[geo_df.index.isin(rgi_ids) & mask]

        hugonnet_dmdtda = selected_gdirs_geo_df['dmdtda'].values * 1000
        hugonnet_err_dmdtda = selected_gdirs_geo_df['err_dmdtda'].values * 1000

        X_labels = ['melt_f', 'prcp_fac', 'temp_bias']
        M = len(X_labels)

        # gdir_hef.settings['error_when_glacier_reaches_boundaries'] = False # TODO- When more realistic, I assume we will not need this?
        distr_fun = st.uniform # Uniform distribution for all parameters
        x_min = np.array([1.5, 0.1, -5.0]) # Minimum values for each parameter
        x_max = np.array([3.0, 6.0, 0.0]) # Maximum values for each parameter

        distr_par = [np.nan] * M
        for i in range(M):
                        distr_par[i] = [x_min[i], x_max[i] - x_min[i]]

        samp_strat = 'lhs'

        N = 100 # Number of samples

        X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N) # Generate the samples, start all with the same initial boundaries

        res_dict = {}

        for i in range(num_of_glaciers):

                # Set these BEFORE calling runoff_execution()
                os.environ["OGGM_CB_GLACIER"] = str(i)
                os.environ["OGGM_CB_TOTAL"]   = str(len(X))
                os.environ["OGGM_CB_UPDATE"]  = "5"
                os.environ["OGGM_CB_START"]   = str(time.time())

                YY = runoff_execution(fun_test = run_with_runoff_for_sa,
                                         X = X, # All samples
                                         gdir = gdirs[i], # Hinteresfirner Glacier directory
                                         years =range(1901, 2020), # years
                                         glacier_index = i,
                                         init_model_yr = 1901, # Simulation start year - needs to be early enough to allow for spinup before the period we are interested in
                                         ys =1901, # Start of the simulation
                                         min_ys = 1901, # Minimum start year
                                         ref_area_yr = rgi_dates[i], # Reference area year - needs to be a year for which we have observed area data for the glacier, so we can use this to constrain the modelled glacier area during the spinup period
                                         spinup_period =95, # Spinup period in years (we are cutting this off, once the glacier has reached an equilibrium state, but this can be changed to a different period if desired)
                                         csv_filepath = str(i)+'_pakistan_runoff_output.csv',
                                         params_csv_filepath=str(i)+'_pakistan_params.csv',
                                         run_task = tasks.run_from_climate_data,
                                         mb_model_method = MultipleFlowlineMassBalance)

                res_dict[i] = YY
                print("DONE Glacier: " + str(i))

if __name__ == "__main__":
        main()
