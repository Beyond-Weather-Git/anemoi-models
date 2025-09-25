import datetime

import numpy as np

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
import torch

import tqdm
import xarray as xr
import glob
import matplotlib.pyplot as plt


multistep_input = 2

name_map_input = {
    # surface vars
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "2m_dewpoint_temperature": "2d",
    "2m_temperature": "2t",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "skin_temperature": "skt",
    "land_sea_mask": "lsm",
    "orography": "z",
    "standard_deviation_of_orography": "sdor",
    "slope_of_sub_gridscale_orography": "slor",
    "total_column_water": "tcw",

    # forcings
    # "cos_latitude": "cos_latitude",
    # "sin_latitude": "sin_latitude",
    # "cos_longitude": "cos_longitude",
    # "sin_longitude": "sin_longitude",
    # "cos_julian_day": "cos_julian_day",
    # "sin_julian_day": "sin_julian_day",
    # "cos_local_time": "cos_local_time",
    # "sin_local_time": "sin_local_time",
    # "insolation": "insolation",
    # pressure-level vars (need suffix)
    "geopotential": "z",
    "temperature": "t",
    "specific_humidity": "q",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    # soil vars
    "soil_temperature_level_1": "stl1",
    "soil_temperature_level_2": "stl2",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",

    # "convective_precipitation": "cp",
    # "total_precipitation": "tp",
    # "100m_u_component_of_wind": "100u",
    # "100m_v_component_of_wind": "100v",
    # "high_cloud_cover":"hcc",
    # "low_cloud_cover":"lcc",
    # "medium_cloud_cover":"mcc",
    # "runoff":"ro",
    # "snowfall":"sf",
    # "surface_solar_radiation_downwards":"ssrd",
    # "surface_thermal_radiation_downwards":"strd",
    # "total_cloud_cover":"tcc"

    
}



def dataset_to_multistep_dict(ds, start_idx, multistep_input, date_coord="time"):
    """Convertit un bloc multistep d'un Dataset en dict {date, fields}"""
    date = str(ds[date_coord].isel(time=start_idx+multistep_input-1).values) 
    fields = {}
    arr_app = []
    for var in ds.data_vars:
        if var not in name_map_input:
            continue
        if "pressure_level" in ds[var].dims:
            for lev_val in ds["pressure_level"].values:
                short_name = name_map_input[var]
                key = f"{short_name}_{lev_val}"
                arr = ds[var].isel(
                    time=slice(start_idx, start_idx + multistep_input)).sel(
                    pressure_level=lev_val).values
                # breakpoint()
                arr = arr.reshape(multistep_input, -1)  
                print("MEAN OF ", key, arr.mean())
                fields[key] = arr.tolist()
                arr_app.append(arr)
        else:
            short_name = name_map_input.get(var, var)
            if "time" not in ds[var].dims:
                arr = ds[var].values
                arr = np.expand_dims(arr, axis=0)
                arr = np.repeat(arr, multistep_input, axis=0)
                print("MEAN OF ", short_name, arr.mean())
                fields[short_name] = arr.tolist()
                arr_app.append(arr)
            else:
                arr = ds[var].isel(
                    time=slice(start_idx, start_idx + multistep_input)
                ).values
                arr = arr.reshape(multistep_input, -1)
                print("MEAN OF ", short_name, arr.mean())
                fields[short_name] = arr.tolist()
                arr_app.append(arr)
    # breakpoint()
    return {"date": date, "fields": fields}

def dataset_to_single_dict(ds, idx, date_coord="time"):
    idx = idx+1
    date = str(ds[date_coord].isel(time=idx).values)
    print("selecting date", date)
    fields = {}

    for var in ds.data_vars:
        if "pressure_level" in ds[var].dims:
            for lev_val in ds["pressure_level"].values:
                short_name = name_map_input[var]
                key = f"{short_name}_{lev_val}"
                arr = ds[var].isel(time=idx).sel(pressure_level=lev_val).values
                fields[key] = arr.reshape(-1).tolist()
        else:
            short_name = name_map_input.get(var, var)
            arr = ds[var].isel(time=idx).values if "time" in ds[var].dims else ds[var].values
            fields[short_name] = arr.reshape(-1).tolist()

    return {
        "date": datetime.datetime.fromisoformat(date.replace("Z", "+00:00")),
        "fields": fields,
    }


def create_ground_truth_dataset(ds):
    """Create ground truth dataset."""
    labels_dicts = []
    for idx in tqdm.tqdm(range(ds.dims["time"]- multistep_input + 1), desc="Creating labels dicts"):
        labels_dicts.append(dataset_to_single_dict(ds, idx))
    print("✅ Created labels dicts with one entry per date")
    return labels_dicts

def create_input_dataset(ds):
    """Create multistep input dataset."""
    all_dicts = []
    for start_idx in tqdm.tqdm(range(ds.dims["time"] - multistep_input + 1), desc="Creating input state dicts"):
        print(start_idx)
        one_dict = dataset_to_multistep_dict(ds, start_idx, multistep_input)
        for k, v in one_dict["fields"].items():
            one_dict["fields"][k] = np.array(v)
        one_dict["date"] = datetime.datetime.fromisoformat(one_dict["date"].replace("Z", "+00:00"))
        all_dicts.append(one_dict)
        if start_idx == 0:
            break
    return all_dicts


def run_inference():
    for input_state in tqdm.tqdm(input_state_bw, desc="Running model and saving npz"):
        for state in runner.run(input_state=input_state, lead_time=12):
            print("computing infefrence for date", state["date"])
            print_state(state)
            print("geo 500 mean", state["fields"]["z_500"].mean())
            np.savez_compressed(f"output_{state['date'].strftime('%Y%m%d%H')}.npz",
                        date=state["date"].isoformat(),
                        **state["fields"])


def load_predictions():
    # --- Load predictions from .npz ---
    pred_files = sorted(glob.glob("output_*.npz"))
    preds = []
    for fname in tqdm.tqdm(pred_files, desc="Loading predictions"):
        pred_npz = np.load(fname, allow_pickle=True)
        date = datetime.datetime.fromisoformat(str(pred_npz["date"]))
        fields = {k: pred_npz[k] for k in pred_npz.files if k != "date"}
        preds.append({"date": date, "fields": fields})

    print(f"✅ Loaded {len(preds)} predictions from .npz")

def compute_rmse(preds, labels_by_date):
    variables = ["2t", "10u", "10v"]
    rmse_by_var = {}

    for var in variables:
        all_sq_errors = []

        for pred in preds:
            date = pred["date"]

            if date not in labels_by_date:
                print(f"⚠️ No label found for {date}, skipping")
                continue

            label = labels_by_date[date]

            # flatten arrays (spatial points)
            pred_vals = np.array(pred["fields"][var], dtype=float).reshape(-1)
            label_vals = np.array(label["fields"][var], dtype=float).reshape(-1)

            # accumulate squared error pointwise
            sq_err = (pred_vals - label_vals) ** 2
            all_sq_errors.append(sq_err)

        if not all_sq_errors:
            print(f"⚠️ No errors accumulated for {var}, skipping")
            continue

        # concat all errors (time × space)
        all_sq_errors = np.concatenate(all_sq_errors)

        # global RMSE
        rmse_global = np.sqrt(all_sq_errors.mean())
        rmse_by_var[var] = rmse_global
        print(f"✅ Global RMSE for {var}: {rmse_global:.4f}")

    # --- Plot all three ---
    lead_times = [6]  # adjust if you have multiple forecast lead times
    plt.figure(figsize=(8, 5))

    for var, rmse_val in rmse_by_var.items():
        plt.plot(lead_times, [rmse_val], marker="o", linestyle="-", label=var)

    plt.xlabel("Lead time (hours)")
    plt.ylabel("RMSE (global, time+space)")
    plt.title("RMSE vs Lead Time (AIFS pipeline, January 2021)")
    plt.grid(True)
    plt.legend()
    plt.savefig("rmse_vs_lead_time.png")
    plt.show()
    plt.close()

def check_all_input_vars(runner, fields: dict[str, np.ndarray]):
    """Checking if all input vars are defined.
    
    Command showing all AIFS variables: runner.checkpoint.typed_variables.
    """
    all_missing_vars = []
    constant_forcings_inputs = []
    for var in runner.checkpoint.typed_variables:
        if runner.checkpoint.typed_variables[var].is_from_input:
            if var not in fields:
                print(f"⚠️ Missing input var: {var}")
                all_missing_vars.append(var)
        else:
            print("✅ Not from input var (remove from name_map_input):", var)            
        if runner.checkpoint.typed_variables[var].is_constant_in_time:
            constant_forcings_inputs.append(var)
    return all_missing_vars, constant_forcings_inputs


ds_path = "/home/ubuntu/bw-dl/data/datasets/processed/era5_aifs-v1_6h_n320_test/test.zarr"
ds = xr.open_zarr(ds_path)
ds_sel = ds.sel(time=slice("2019-01-30", "2019-01-31"))

print(ds_sel.time.values)
# labels_dict = create_ground_truth_dataset(ds_sel)
input_state_bw = create_input_dataset(ds_sel)
print("✅ Created input state dict")
checkpoint = {"huggingface":"ecmwf/aifs-single-1.0"}
runner = SimpleRunner(checkpoint, device="cuda")
# breakpoint()
missing_vars, forcing_vars = check_all_input_vars(runner, input_state_bw[0]["fields"]) 
runner.constant_forcings_inputs = runner.checkpoint.constant_forcings_inputs(runner, input_state_bw[0])
runner.dynamic_forcings_inputs = runner.checkpoint.dynamic_forcings_inputs(runner, input_state_bw[0])
runner.boundary_forcings_inputs = runner.checkpoint.boundary_forcings_inputs(runner, input_state_bw[0])
normalized = runner.prepare_input_tensor(input_state_bw[0])
run_inference()
print("all done")
breakpoint()
# preds = load_predictions()
# labels_by_date = {entry["date"]: entry for entry in labels_dicts}
# compute_rmse(preds, labels_by_date)






# also save labels dict as npz
# labels_npz = {}

# for i, label in enumerate(labels_dicts):
#     prefix = f"t{i:03d}"  # prefix to distinguish each timestep
#     labels_npz[f"{prefix}_date"] = str(label["date"])  # save as string
#     for k, v in label["fields"].items():
#         labels_npz[f"{prefix}_{k}"] = np.array(v)

# np.savez_compressed("labels.npz", **labels_npz)
# print("✅ Saved labels as labels.npz")

# --- Load labels from labels.npz ---
# labels_npz = np.load("labels.npz", allow_pickle=True)

# labels_by_date = {}
# # labels were saved with keys like t000_date, t000_z_500, ...
# n_labels = len([k for k in labels_npz.keys() if k.endswith("_date")])
# for i in range(n_labels):
#     prefix = f"t{i:03d}"
#     date = datetime.datetime.fromisoformat(str(labels_npz[f"{prefix}_date"]))
#     fields = {
#         k.replace(f"{prefix}_", ""): labels_npz[k]
#         for k in labels_npz.keys()
#         if k.startswith(prefix) and not k.endswith("_date")
#     }
#     labels_by_date[date] = {"date": date, "fields": fields}

# print(f"✅ Loaded {len(labels_by_date)} labels from labels.npz")