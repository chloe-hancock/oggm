from oggm import entity_task
import logging 
import numpy as np
from oggm.core.flowline import run_with_hydro
from oggm.core.massbalance import MonthlyTIModel
import oggm.cfg as cfg
import pandas as pd
import xarray as xr

# Module logger
log = logging.getLogger(__name__)

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
                    run_task=None,
                    mb_model_method=None,
                    save_output= True):
    """
    # TODO: Update the model inputs here and add this into the notebook?
    # TODO: Remove some of the inputs and workflow? So this is more similar to the run_with_hydro task.
    Calculates the runoff from a glacier using the `run_with_hydro` task, and saves the outputs to a CSV file.
    This 
    
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
    
    mbdf = gdir.get_ref_mb_data().loc[years] # WGMS data for the glacier
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

    # Model output years
    hydro_years = ds['time'].values

    y1 = int(max(hydro_years.min(), years[0] + spinup_period))
    y2 = int(min(hydro_years.max(), years[-1]))

    if y1 > y2:
        log.warning(f"No valid hydrological years for parameters {mb_params}")
        return None
    
    df_area = ds['area_m2'].loc[y1:y2].values

    df_volume = ds['volume_m3'].loc[y1:y2].values

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
            'runoff': runoff,
            'mass_balance': df_mb,
            'area': df_area,
            'volume': df_volume})
    
    if save_output:
        df.to_csv(cfg.PATHS['working_dir'] + '/' + str(row_index) + '_' + csv_filepath, index=False)

    return np.array(runoff)