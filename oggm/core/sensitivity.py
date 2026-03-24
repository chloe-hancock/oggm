from oggm import entity_task
import logging 
import numpy as np
from oggm.core.flowline import run_with_hydro
from oggm.core.massbalance import MonthlyTIModel
import oggm.cfg as cfg
import pandas as pd
import xarray as xr
import numpy as np
import pandas as pd
from oggm import cfg, workflow
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# Module logger
log = logging.getLogger(__name__)

#######################################################################
# Function for calculating the runoff outputs for Sensitivity Analysis
#######################################################################

@entity_task(log)
def run_with_runoff_for_sa(gdir, *,
                    mb_params=None,
                    row_index=None,
                    years=None,
                    init_model_yr=None,
                    ys=None,
                    min_ys=None,
                    ref_area_yr=None,
                    spinup_period=None,
                    settings_filesuffix='',
                    csv_filepath='runoff_output.csv',
                    params_csv_filepath='params.csv',
                    run_task=None,
                    mb_model_method=None,
                    save_output= True,
                    progress_callback=None):
    """
    # TODO: Update the model inputs here and add this into the notebook?
    # TODO: Remove some of the inputs and workflow? So this is more similar to the run_with_hydro task.
    Calculates the runoff from a glacier using the `run_with_hydro` task, and outputs the timeseries of 
    annual runoff. This also write the following outputs to a CSV: the time series for annual mass balance, 
    time series for annual runoff, the annual area, the annual volume and the years that we are investigating.
    
    mb_params: tuple
        The mass balance parameters to use for the run, 
        in the order of melt_f, prcp_fac, temp_bias
    row_index: int
        The index of the row in the input parameter dataframe, used to create a unique identifier for the output file. 
        This is included to allow for parallel runs with different parameters, where each run can be identified by its row index in the input dataframe.
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    years: range or list
        The years we will be using for analysis, which should be a subset of the years for which we run the model. 
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    ys : int
        start year of the model run (needs to be set)
    min_ys : int
        if you want to impose a minimum start year, regardless if the glacier
        inventory date is earlier (e.g. if climate data does not reach).
    ref_area_yr : int
        the hydrological output is computed over a reference area, which
        per default is the largest area covered by the glacier in the simulation
        period. Use this kwarg to force a specific area to the state of the
        glacier at the provided simulation year.
    spinup_period: int
        The number of years to run the model in spinup mode before starting the 
        main simulation.
    settings_filesuffix : str
        a filesuffix for using a specific settings file
    csv_filepath: str
        The path to the CSV file where the output will be saved.
    run_task : func
        any of the `run_*`` tasks in the oggm.flowline module.
        The mass balance model used needs to have the `add_climate` output
        kwarg available though.
    save_output: bool
        Whether to save the output to a CSV file or not. Default is True.
    """
    try:
        mbdf = gdir.get_ref_mb_data().loc[years] # WGMS data for the glacier
    except (RuntimeError):
    # If no WGMS data available create an empty frame with the right index
        mbdf = pd.DataFrame(index=years)
    gdir.settings['error_when_glacier_reaches_boundaries'] = False # TODO- When more realistic, I assume we will not need this?

    # Set the parameter values
    melt_f, prcp_fac, temp_bias = mb_params

    # TODO: Can use the other style of massbalance model? Perhaps change this?
    mb = mb_model_method(
        gdir,
        mb_model_class=MonthlyTIModel,
        melt_f=float(melt_f),
        prcp_fac=float(prcp_fac),
        temp_bias=float(temp_bias),
        check_calib_params=False,
        ) 

    
    fls = gdir.read_pickle('inversion_flowlines') # Read flowlines
    mbdf['mod_mb'] = mb.get_specific_mb(fls=fls, year=mbdf.index) # Compute modelled mass balance  

    # Create unique file identifier based on parameters, where the model output is saved
    file_id = f'_hydro_mf{melt_f:.2f}_pf{prcp_fac:.2f}_tb{temp_bias:.2f}'
    
    # Uses run with hydro to calculate hydrological output, so we can calculate the runoff
    run_with_hydro(
        gdir,
        run_task=run_task,
        ys=ys, # The simulation start year
        min_ys=min_ys, # For the run from climate data, to ensure we have data from 1979
        init_model_yr=init_model_yr,
        ref_area_yr=ref_area_yr,
        mb_model=mb, # The modified MB model
        store_monthly_hydro=True,
        output_filesuffix=file_id,
        settings_filesuffix= settings_filesuffix # TODO: Check how to use this with the rest of the code?
    )

    with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=file_id)) as ds:
        # The last step of hydrological output is NaN (we can't compute it for this year)
        ds = ds.isel(time=slice(0, -1)).load()
    

    # These summed variabels give the total runoff from the glacier
    runoff_vars = ['melt_off_glacier', 'melt_on_glacier','liq_prcp_off_glacier', 'liq_prcp_on_glacier']

    y1 = years[0] + spinup_period
    y2 = years[-1]

    if y1 > y2:
        log.warning(f"No valid hydrological years for parameters {mb_params}")
        return None
    
    df_area = ds['area_m2'].loc[y1:y2].values * 1e-6

    df_volume = ds['volume_m3'].loc[y1:y2].values * 1e-9

    # Convert MB index to integer-year
    mbdf_annual = mbdf.loc[y1:y2].copy()
    mbdf_annual.index = mbdf_annual.index.astype(int)
    df_mb = mbdf_annual['mod_mb'].values

    # Extract the relevant runoff variables
    df_annual = ds[runoff_vars].to_dataframe()

    # Convert runoff from kg → Mt-equivalent and sum components
    df_runoff = df_annual.sum(axis = 1) * 1e-9
    runoff = df_runoff.loc[y1:y2].values

    # Write the output to a csv file
    df = pd.DataFrame({
            'years': list(range(y1,y2+1)),
            'runoff_Mt': runoff,
            'mass_balance': df_mb,
            'area_km2': df_area,
            'volume_km3': df_volume})
    
    param_df = pd.DataFrame({"params": mb_params})
    
    if save_output:
        df.to_csv(cfg.PATHS['working_dir'] + '/' + str(row_index) + '_' + csv_filepath, index=False)
        param_df.to_csv(cfg.PATHS['working_dir'] + '/' + str(row_index) + '_' + params_csv_filepath, index=False)
    
    if progress_callback is not None:
        progress_callback()

    return np.array(runoff)

#######################################################################
# Function for Reducing Bounds - Check using Linear Regression Method
#######################################################################
def parameter_bounding(
    X,
    y_area,
    y_mass_balance,
    hugonnet,
    hugonnet_error,
    obs_area,
    year_idx,
    area_percentile=10,
    area_bounding_flag=True,
    hugonnet_bounding_flag=True
):
    """
    Selects parameter sets based on:
    - Deviation from area (y_area close to 0)
    - Hugonnet mass-balance uncertainty bounds (y_hugonnet within upper/lower error)
    """

    # Check flags 
    if not area_bounding_flag and not hugonnet_bounding_flag:
        raise ValueError("At least one bounding method must be chosen!")
    
    lower_error = hugonnet - hugonnet_error
    upper_error = hugonnet + hugonnet_error

    if hugonnet_bounding_flag:
        if lower_error is None or upper_error is None:
            raise ValueError("lower_error and upper_error must be provided for Hugonnet bounding.")
        if lower_error >= upper_error:
            raise ValueError("lower_error must be < upper_error.")
        
    # Prepare arrays 
    y_area = np.asarray(y_area)
    y_mass_balance = np.asarray(y_mass_balance)

    # Areas at the RGI year, before and after
    area_at_rgi_yr = []
    area_before_rgi_yr = []
    area_after_rgi_yr = []

    for i in range(len(X)):
        area_at_rgi_yr.append(y_area[i][year_idx]) # year at RGI Observation
        area_before_rgi_yr.append(y_area[i][year_idx-1]) # year before RGI Observation
        area_after_rgi_yr.append(y_area[i][year_idx+1]) # year after RGI Observation

    # 2. Area bias threshold 
    area_lower_bound = obs_area * (1 - area_percentile/100)
    area_upper_bound = obs_area * (1 + area_percentile/100)

    # Build mask 
    # Start with everything selected
    mask = np.ones(len(y_area), dtype=bool)

    area_at_rgi_yr = np.array(area_at_rgi_yr)
    area_before_rgi_yr = np.array(area_before_rgi_yr)
    area_after_rgi_yr = np.array(area_after_rgi_yr)

    if area_bounding_flag:
        cond_at = (area_at_rgi_yr <= area_upper_bound) & (area_at_rgi_yr >= area_lower_bound)
        cond_before = (area_before_rgi_yr <= area_upper_bound) & (area_before_rgi_yr >= area_lower_bound)
        cond_after = (area_after_rgi_yr <= area_upper_bound) & (area_after_rgi_yr >= area_lower_bound)
        
        ok_area = (cond_at | cond_before | cond_after)
        mask &= ok_area

        kept = mask.sum()
        pct  = (kept / len(y_area)) * 100
        print(f"After area bounding: {kept}/{len(y_area)} values remain ({pct:.1f}%)")

    if hugonnet_bounding_flag:
        before = mask.sum()  # how many were left before this step
        mask &= (y_mass_balance >= lower_error) & (y_mass_balance <= upper_error)
        kept = mask.sum()
        pct  = (kept / len(y_area)) * 100
        step_pct = (kept / before) * 100 if before > 0 else 0
        print(f"After Hugonnet bounding: {kept}/{len(y_area)} values remain "
              f"({pct:.1f}% of original; {step_pct:.1f}% kept from previous step)")

    # 4. Apply mask 
    if mask.sum() == 0:
        raise ValueError("No samples satisfy the selected bounds. "
                         "Try relaxing percentile or Hugonnet range.")

    good_X = X[mask]

    print("There are " + str(len(good_X)) + " samples captured within the given bounds.")

    lb = good_X.min(axis=0)
    ub = good_X.max(axis=0)

    return lb, ub, good_X

#######################################################################
# Metric calculator for hydro outputs - for Sensitivity Analysis
#######################################################################
def hydro_output_metric_calculator(runoff):
    '''
    Calculates the output metrics for runoff, the mean and the standard deviation. 

    This is currently only used for calculting the mean and standard deviation of the runoff
    but can be added to in the future.

    Parameters:
    ------------

    runoff: np.array
        The runoff time series generated from the oggm_sim method, when the output is the runoff time series.

    Return:
    ------------

    YY: np.array
        An array containing the values of:
            - The annual runoff mean.
            - The annual runoff standard deviation.
    '''
    YY = np.nan * np.ones((len(runoff), 2))

    for i, x in enumerate(runoff):
        YY[i,0] = np.mean(x)
        YY[i,1] = np.std(x)

    return YY

#######################################################################
# Execute the runoff, both sequentially and in multiprocessing using the 
#######################################################################
def runoff_execution(fun_test, X, gdir,
                        years, init_model_yr, ys, min_ys,
                        ref_area_yr, spinup_period,
                        csv_filepath, params_csv_filepath, run_task, mb_model_method):

    all_experiments = []

    # PROGRESS BAR FOR SAMPLES
    pbar = tqdm(total=len(X), desc="Processing samples", unit="sample", dynamic_ncols=True)

    # Callback used by each entity task
    def update_pbar():
        pbar.update(1)
        pbar.refresh()   # fix buffering on HPC

    # Shared parameters for each sample
    common = dict(
        years=years,
        init_model_yr=init_model_yr,
        ys=ys,
        min_ys=min_ys,
        ref_area_yr=ref_area_yr,
        spinup_period=spinup_period,
        csv_filepath=csv_filepath,
        params_csv_filepath=params_csv_filepath,
        run_task=run_task,
        mb_model_method=mb_model_method
    )

    # Build experiment list
    for i, sample_row in enumerate(X):
        kw = dict(common)
        kw.update(
            mb_params=sample_row,
            row_index=i,
            settings_filesuffix=f"_exp{i}",
            progress_callback=update_pbar
        )
        all_experiments.append((gdir, kw))

    # Run experiments
    old_continue_one_error = cfg.PARAMS["continue_on_error"]
    cfg.PARAMS["continue_on_error"] = True
    out_list = workflow.execute_entity_task(fun_test, all_experiments)
    cfg.PARAMS["continue_on_error"] = old_continue_one_error

    # Finish progress bar
    pbar.close()

    return hydro_output_metric_calculator(out_list)


#######################################################################
# The goodness of fit functions for the mass balance values 
#######################################################################
def mean_diff(x, y):
    output = abs(np.mean(x) - np.mean(y))
    return output

def std_diff(x, y):
    output = abs(np.std(x) - np.std(y))
    return output

@entity_task(log)
def spinup_area_volume(gdir, *,
                    mb_params=None,
                    years=None,
                    init_model_yr=None,
                    ys=None,
                    min_ys=None,
                    ref_area_yr=None,
                    settings_filesuffix='',
                    run_task=None,
                    mb_model_method=None):
    """
    Calculates the area and volume from a glacier using the `run_with_hydro` task, and outputs the timeseries of 
    annual runoff for the full period (including the spinup time we are using in the run_with_runoff_for_sa method.)
    
    mb_params: tuple
        The mass balance parameters to use for the run, 
        in the order of melt_f, prcp_fac, temp_bias
    row_index: int
        The index of the row in the input parameter dataframe, used to create a unique identifier for the output file. 
        This is included to allow for parallel runs with different parameters, where each run can be identified by its row index in the input dataframe.
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    years: range or list
        The years we will be using for analysis, which should be a subset of the years for which we run the model. 
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    ys : int
        start year of the model run (needs to be set)
    min_ys : int
        if you want to impose a minimum start year, regardless if the glacier
        inventory date is earlier (e.g. if climate data does not reach).
    ref_area_yr : int
        the hydrological output is computed over a reference area, which
        per default is the largest area covered by the glacier in the simulation
        period. Use this kwarg to force a specific area to the state of the
        glacier at the provided simulation year.
    spinup_period: int
        The number of years to run the model in spinup mode before starting the 
        main simulation.
    settings_filesuffix : str
        a filesuffix for using a specific settings file
    csv_filepath: str
        The path to the CSV file where the output will be saved.
    run_task : func
        any of the `run_*`` tasks in the oggm.flowline module.
        The mass balance model used needs to have the `add_climate` output
        kwarg available though.
    save_output: bool
        Whether to save the output to a CSV file or not. Default is True.
    """
    try:
        mbdf = gdir.get_ref_mb_data().loc[years] # WGMS data for the glacier
    except (RuntimeError):
    # If no WGMS data available create an empty frame with the right index
        mbdf = pd.DataFrame(index=years)

    gdir.settings['error_when_glacier_reaches_boundaries'] = False # TODO- When more realistic, I assume we will not need this?
    
    # Set the parameter values
    melt_f, prcp_fac, temp_bias = mb_params

    # TODO: Can use the other style of massbalance model? Perhaps change this?
    mb = mb_model_method(
        gdir,
        mb_model_class=MonthlyTIModel,
        melt_f=float(melt_f),
        prcp_fac=float(prcp_fac),
        temp_bias=float(temp_bias),
        check_calib_params=False,
        ) 
    
    fls = gdir.read_pickle('inversion_flowlines') # Read flowlines
    mbdf['mod_mb'] = mb.get_specific_mb(fls=fls, year=mbdf.index) # Compute modelled mass balance  

    # Create unique file identifier based on parameters, where the model output is saved
    file_id = f'_hydro_mf{melt_f:.2f}_pf{prcp_fac:.2f}_tb{temp_bias:.2f}'
    
    # Uses run with hydro to calculate hydrological output, so we can calculate the runoff
    run_with_hydro(
        gdir,
        run_task=run_task,
        ys=ys, # The simulation start year
        min_ys=min_ys, # For the run from climate data, to ensure we have data from 1979
        init_model_yr=init_model_yr,
        ref_area_yr=ref_area_yr,
        mb_model=mb, # The modified MB model
        store_monthly_hydro=True,
        output_filesuffix=file_id,
        settings_filesuffix= settings_filesuffix # TODO: Check how to use this with the rest of the code?
    )

    with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=file_id)) as ds:
        # The last step of hydrological output is NaN (we can't compute it for this year)
        ds = ds.isel(time=slice(0, -1)).load()
    
    area = ds['area_km2'].values * 1e-6

    volume = ds['volume_km3'].values * 1e-9

        # Write the output to a csv file
    df = pd.DataFrame({
            'area_km2': area,
            'volume_km3': volume})

    return df

#######################################################################
# Execute the runoff, both sequentially and in multiprocessing using the 
#######################################################################
def runoff_execution_full_spinup(fun_test, X, gdir,
                        years, init_model_yr, ys, min_ys,
                        ref_area_yr, spinup_period,
                        csv_filepath, params_csv_filepath, run_task, mb_model_method):

    all_experiments = []

    # Shared parameters (same for all experiments)
    common = dict(
        years=years,
        init_model_yr=init_model_yr,
        ys=ys,
        min_ys=min_ys,
        ref_area_yr=ref_area_yr,
        spinup_period=spinup_period,
        csv_filepath=csv_filepath,
        params_csv_filepath=params_csv_filepath,
        run_task=run_task,
        mb_model_method=mb_model_method,
    )

    # One experiment per sample
    for i, sample_row in enumerate(X):
        kw = dict(common)
        kw.update(
            mb_params=sample_row,
            row_index=i,
            settings_filesuffix=f"_exp{i}"   # <- writing a new settings_filesuffix with each sample
        )
        all_experiments.append((gdir, kw))
    
    old_continue_one_error = cfg.PARAMS["continue_on_error"]
    cfg.PARAMS["continue_on_error"] = True
    # Run all experiments in parallel
    out_list = workflow.execute_entity_task(fun_test, all_experiments)
    print(type(out_list), len(out_list))
    cfg.PARAMS["continue_on_error"] = old_continue_one_error

    return out_list

def colour_plotting_timeseries(j, 
             plot_area_flag=True, 
             plot_spec_mb_flag=False, 
             plot_runoff= True, 
             plot_mass_balance=False,
             years_dict= None,
             rgi_dates = None,
             area_dict = None,
             rgi_area_km2s = None,
             mass_balance_dict = None,
             hugonnet_dmdtda = None,
             hugonnet_dmdtda_err = None,
             N = None,
             runoff_dict = None):
    
    if (plot_area_flag is False) and (plot_spec_mb_flag is False):
        raise ValueError("One of the success index flags must be set to True!")
    
    if (plot_area_flag is False) and (plot_spec_mb_flag is False):
        raise ValueError("One of the plotting flags must be set to True!")
    
    if (plot_area_flag is True) and (plot_spec_mb_flag is True):
        raise ValueError("Only one of the success index flags must be set to True!")
    
    if (plot_area_flag is True) and (plot_spec_mb_flag is True):
        raise ValueError("Only one of the plotting flags must be set to True!")

    ##############################################################################################
    # The success measure index is now set to the Area Bias at the Observation Year
    ##############################################################################################

    if plot_area_flag == True:
        index = np.where(years_dict[j][0] == rgi_dates[j])[0][0]

        new_area_in_rgi_year = []
        for new_area_sample in area_dict[j]:
            new_area_in_rgi_year.append(new_area_sample[index])

        success_measure_index = []

        for i in range(len(new_area_in_rgi_year)):
            success_measure_index.append((new_area_in_rgi_year[i] - rgi_area_km2s[j]))

    ##############################################################################################
    # The success measure index is now set to the Specific Mass Balance in the Observation Period
    ##############################################################################################

    if plot_spec_mb_flag == True:

        years = np.array(years_dict[j][0])
        idx = np.where((years >= 2000) & (years <= 2020))[0]

        # mass balance ensemble is a list of arrays
        mb_list = mass_balance_dict[j]
        mb_means = []

        for sample in mb_list:
            sample = np.array(sample)
            sliced = sample[idx] # slice 2000–2019
            mb_means.append(sliced.mean())
        
        success_measure_index = []
        for i in range(len(mb_means)):
            success_measure_index.append(mb_means[i])

    # compile csvs to plot timeseries and plot to view the mass balance time series for each of the 50 samples, to see how they are looking and check that they make sense before we calculate the sensitivity indices
    # Colormap and normalization
    cmap = plt.cm.coolwarm

    vmin = min(success_measure_index)
    vmax = max(success_measure_index)
    norm_success_measure_index = plt.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(15,5))
    fig, ax = plt.subplots(figsize=(15, 5))

    for i in range(N):
        color = cmap(norm_success_measure_index(success_measure_index[i]))
        if plot_runoff is True:
            plt.plot(years_dict[j][i], runoff_dict[j][i], color=color, linewidth=0.5)
            plt.title('Runoff time series for each parameter sample, N = %d' % N)
        if plot_mass_balance is True:
            plt.plot(years_dict[j][i], mass_balance_dict[j][i], color=color, linewidth=0.5)
            
    if plot_mass_balance is True:
        plt.axhline(hugonnet_dmdtda[j], color='teal')
        plt.axhline(hugonnet_dmdtda[j] + hugonnet_dmdtda_err[j], linestyle = '--', color='teal')
        plt.axhline(hugonnet_dmdtda[j] - hugonnet_dmdtda_err[j], linestyle = '--', color='teal')
        plt.axhspan(hugonnet_dmdtda[j] - hugonnet_dmdtda_err[j], hugonnet_dmdtda[j] + hugonnet_dmdtda_err[j], color='teal', alpha = 0.25, label = 'Hugonnet Observation and Error', zorder=3)
        plt.legend()
        plt.title('Mass Balance time series for each parameter sample, N = %d' % N)

    # Add colorbar linked to the same colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_success_measure_index)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm, ax=ax)

    if plot_area_flag == True:
        cbar.set_label('Area Bias at RGI Year')
    elif plot_spec_mb_flag == True:
        cbar.set_label('Specific Mass Balance in 2000-2020')

    plt.xlabel('Years')
    if plot_runoff is True:
        plt.ylabel('Runoff')
    if plot_mass_balance is True:
        plt.ylabel("Mass Balance")
    plt.show()
