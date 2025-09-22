# NOT WORKING

from anemoi.training.train.train import AnemoiTrainer
ds_path = "/Users/semv/Downloads/era5_aifs-v1_6h_n320_test/test.zarr"
# ds = xr.open_zarr(ds_path)
from omegaconf import OmegaConf
aifs_config = OmegaConf.load("config_finetuning.yaml")


aifs_config.hardware.paths.data = ds_path
aifs_config.hardware.paths.output = "/path/to/your/output"
aifs_config.training.fork_run_id = ""  # or actual run ID

aifs_config.config_validation = True
trainer = AnemoiTrainer(aifs_config)