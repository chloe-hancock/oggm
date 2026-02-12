from oggm import cfg, tasks
from oggm.core.massbalance import MultipleFlowlineMassBalance
from oggm.core import massbalance
import numpy as np
import xarray as xr
import os

def oggm_sim(gdir, melt_f, prcp_fac, temp_bias, years=range(1979,2020), run_with_hydro=True):
    # OGGM config (per worker)
    cfg.initialize()

    # Read MB observations
    mbdf = gdir.get_ref_mb_data().loc[years]

    # Set MB parameters
    cfg.PARAMS['melt_f'] = melt_f
    cfg.PARAMS['prcp_fac'] = prcp_fac
    cfg.PARAMS['temp_bias'] = temp_bias

    # Build MB model
    mb = MultipleFlowlineMassBalance(
        gdir,
        mb_model_class=massbalance.MonthlyTIModel,
        melt_f=melt_f, prcp_fac=prcp_fac, temp_bias=temp_bias,
        check_calib_params=False
    )

    # Compute modeled MB
    fls = gdir.read_pickle('inversion_flowlines')
    mbdf['mod_mb'] = mb.get_specific_mb(fls=fls, year=mbdf.index)

    if not run_with_hydro:
        return np.array(mbdf['mod_mb']), None

    # Unique suffix
    suffix = f"_mf{melt_f:.2f}_pf{prcp_fac:.2f}_tb{temp_bias:.2f}"

    # Run spinup
    tasks.run_dynamic_spinup(
        gdir, mb_model_historical=mb,
        spinup_start_yr=1990, spinup_period=25, min_spinup_period=10,
        output_filesuffix="_spinup"
    )

    # Run hydro
    tasks.run_with_hydro(
        gdir,
        run_task=tasks.run_from_climate_data,
        input_model_filesuffix="_spinup",
        ys=1979, min_ys=1979,
        fixed_geometry_spinup_yr=1979,
        mb_model=mb,
        store_monthly_hydro=True,
        output_filesuffix=suffix
    )

    # Read output
    with xr.open_dataset(gdir.get_filepath("model_diagnostics", filesuffix=suffix)) as ds:
        ds = ds.isel(time=slice(0, -1)).load()

    runoff_vars = ['melt_off_glacier','melt_on_glacier','liq_prcp_off_glacier','liq_prcp_on_glacier']
    runoff = (ds[runoff_vars].to_dataframe().clip(0) * 1e-9).sum(axis=1).to_numpy()

    return np.array(mbdf['mod_mb']), runoff