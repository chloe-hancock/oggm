from html import parser
import os
import numpy as np
import scipy.stats as st

from oggm import cfg, workflow, utils
from oggm import tasks
from oggm.core.massbalance import MultipleFlowlineMassBalance

from safepython.sampling import AAT_sampling # Functions to perform the input sampling
import os
import time

from oggm.core.sensitivity import run_with_runoff_for_sa, runoff_execution

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
                        help="Maximum values for each parameter")

    parser.add_argument("--x_min", type=str, required=True,
                        help="Minimum values for each parameter")
    
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for results (optional)")

    return parser.parse_args()

def main():

        args = get_args()
        cfg.initialize(logging_level="CRITICAL") # To suppress OGGM logging output during the sensitivity analysis runs
        cfg.PATHS['working_dir'] = args.work_dir
        cfg.PARAMS['store_model_geometry'] = True
        cfg.PARAMS['min_ice_thick_for_length'] = 1
        rgi_ids = args.rgi_ids

        cfg.PARAMS['use_multiprocessing'] = True  # To speed up sensitivity analysis runs

        # We pick the elevation-bands glaciers
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup'
        gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=4, prepro_border=160, prepro_base_url=base_url)
        # Get the Hugonnet mass balance and set up dataframe
        geo_df = utils.get_geodetic_mb_dataframe()
        geo_df.loc[rgi_ids]

        # Hydrological model workflow steps before running with hydro

        num_of_glaciers = len(gdirs)

        rgi_dates = []
        rgi_area_km2s = []

        for gdir in gdirs:
                rgi_dates.append(gdir.rgi_date)
                rgi_area_km2s.append(gdir.rgi_area_km2)

                # And match the Hugonnet
        geo_df = utils.get_geodetic_mb_dataframe()

        X_labels = ['melt_f', 'prcp_fac', 'temp_bias']
        M = len(X_labels)

        distr_fun = st.uniform # Uniform distribution for all parameters
        
        x_max = [float(v) for v in args.x_max.split()]
        x_min = [float(v) for v in args.x_min.split()]

        distr_par = [np.nan] * M
        for i in range(M):
                        distr_par[i] = [x_min[i], x_max[i] - x_min[i]]

        samp_strat = 'lhs'

        N = args.N # Number of samples

        X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N) # Generate the samples, start all with the same initial boundaries
        res_dict = {}

        for i in range(num_of_glaciers):
                # Set these before calling runoff_execution
                os.environ["OGGM_CB_GLACIER"] = str(i)
                os.environ["OGGM_CB_TOTAL"]   = str(len(X))
                os.environ["OGGM_CB_UPDATE"]  = "5"
                os.environ["OGGM_CB_START"]   = str(time.time())

                YY = runoff_execution(fun_test = run_with_runoff_for_sa,
                                         X = X, # All samples
                                         gdir = gdirs[i], # Hinteresfirner Glacier directory
                                         years =range(1901, 2020), # years
                                         init_model_yr = 1901, # Simulation start year - needs to be early enough to allow for spinup before the period we are interested in
                                         ys =1901, # Start of the simulation
                                         min_ys = 1901, # Minimum start year
                                         ref_area_yr = rgi_dates[i], # Reference area year - needs to be a year for which we have observed area data for the glacier, so we can use this to constrain the modelled glacier area during the spinup period
                                         spinup_period =95, # Spinup period in years (we are cutting this off, once the glacier has reached an equilibrium state, but this can be changed to a different period if desired)
                                         out_dir = args.out_dir, # Output directory for results
                                         csv_filepath= args.output_csv_path,
                                         params_csv_filepath= args.params_csv_path,
                                         run_task = tasks.run_from_climate_data,
                                         mb_model_method = MultipleFlowlineMassBalance)

                res_dict[i] = YY

if __name__ == "__main__":
        main()
