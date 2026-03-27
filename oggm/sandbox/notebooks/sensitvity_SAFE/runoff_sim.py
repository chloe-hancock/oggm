import os
import numpy as np
import scipy.stats as st

from oggm import cfg, workflow, utils
from oggm import tasks
from oggm.core.massbalance import MultipleFlowlineMassBalance

from safepython.sampling import AAT_sampling # Functions to perform the input sampling
import os
import time

from oggm.core.sensitivity import run_with_runoff_for_sa, runoff_execution, compile_faulty_rows

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Run OGGM glacier sensitivity analysis"
    )

    parser.add_argument("--work_dir", type=str, required=True,
                        help="Path to OGGM working directory")

    parser.add_argument("--border", type=float, required=True,
                        help="Buffer size around glacier geometries")

    parser.add_argument("--store_model_geom", type=bool, required=True,
                        help="Whether to store model geometry (True/False)")

    parser.add_argument("--min_ice_thick", type=float, required=True,
                        help="Minimum ice thickness for length computation")

    parser.add_argument("--rgi_ids", type=str, required=True,
                        help="RGI IDs to process (space-separated or single string)")

    parser.add_argument("--multi_process", type=bool, required=True,
                        help="Whether to use multiprocessing (True/False)")

    parser.add_argument("--base_url", type=str, required=True,
                        help="Base URL or directory containing glacier inputs")

    parser.add_argument("--N", type=int, required=True,
                        help="Number of samples")

    parser.add_argument("--year_start", type=int, required=True,
                        help="Simulation start year")
    
    parser.add_argument("--year_end", type=int, required=True,
                        help="Simulation end year")
    
    parser.add_argument("--spinup_period", type=int, required=True,
                        help="Numbe of years simulation is spun-up for")

    parser.add_argument("--output_csv_path", type=str, required=True,
                        help="Output CSV paths")
    
    parser.add_argument("--params_csv_path", type=str, required=True,
                        help="Parameters CSV paths")
    return parser.parse_args()

def main():

        args = get_args()
        cfg.initialize(logging_level='CRITICAL')
        cfg.PATHS['working_dir'] = args.work_dir
        cfg.PARAMS['store_model_geometry'] = args.store_model_geom
        cfg.PARAMS['min_ice_thick_for_length'] = args.min_ice_thick
        rgi_ids = [args.rgi_ids]

        cfg.PARAMS['use_multiprocessing'] = args.multi_process  # To speed up sensitivity analysis runs

        # We pick the elevation-bands glaciers because they run a bit faster - but they create more step changes in the area outputs
        base_url = args.base_url
        gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=4, prepro_border=160, prepro_base_url=base_url)
        # gdirs = workflow.init_glacier_directories(rgi_ids)

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
        x_min = np.array([1.5, 1.0, -5.0]) # Minimum values for each parameter
        x_max = np.array([3.0, 6.0, 0.0]) # Maximum values for each parameter

        distr_par = [np.nan] * M
        for i in range(M):
                        distr_par[i] = [x_min[i], x_max[i] - x_min[i]]

        samp_strat = 'lhs'

        N = args.N # Number of samples

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
                                         years =range(args.year_start, args.year_end), # years
                                         glacier_index = i,
                                         init_model_yr = args.year_start, # Simulation start year - needs to be early enough to allow for spinup before the period we are interested in
                                         ys =args.year_start, # Start of the simulation
                                         min_ys = args.year_start, # Minimum start year
                                         ref_area_yr = rgi_dates[i], # Reference area year - needs to be a year for which we have observed area data for the glacier, so we can use this to constrain the modelled glacier area during the spinup period
                                         spinup_period =args.spinup_period, # Spinup period in years (we are cutting this off, once the glacier has reached an equilibrium state, but this can be changed to a different period if desired)
                                         csv_filepath = str(i)+args.output_csv_path,
                                         params_csv_filepath=str(i)+args.params_csv_path,
                                         run_task = tasks.run_from_climate_data,
                                         mb_model_method = MultipleFlowlineMassBalance)

                res_dict[i] = YY
                print("DONE Glacier: " + str(i))

                print("The faulty rows", compile_faulty_rows(YY, X))

if __name__ == "__main__":
        main()
