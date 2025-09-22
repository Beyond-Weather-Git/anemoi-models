# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import uuid

import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DotDict
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    multi_step : bool
        Whether the model uses multi-step input.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    supporting_arrays : dict
        Numpy arraysto store in the checkpoint.
    data_indices : dict
        Indices for the data.
    pre_processors : Processors
        Pre-processing steps to apply to the data before passing it to the model.
    post_processors : Processors
        Post-processing steps to apply to the model's output.
    model : AnemoiModelEncProcDec
        The underlying Anemoi model.
    """

    def __init__(
        self,
        *,
        config: DotDict,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        supporting_arrays: dict = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays if supporting_arrays is not None else {}
        self.data_indices = data_indices
        self._build_model()


    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Instantiate processors
        processors = [
            [name, instantiate(processor, data_indices=self.data_indices, statistics=self.statistics)]
            for name, processor in self.config.data.processors.items()
        ]

        # Assign the processor list pre- and post-processors
        self.pre_processors = Processors(processors)
        self.post_processors = Processors(processors, inverse=True)

        # Instantiate the model
        self.model = instantiate(
            self.config.model.model,
            model_config=self.config,
            data_indices=self.data_indices,
            graph_data=self.graph_data,
            _recursive_=False,  # Disables recursive instantiation by Hydra
        )

        # Use the forward method of the model directly
        self.forward = self.model.forward

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        from pathlib import Path
        import os
        import numpy as np
        import uuid
        import torch
        import json
        import re

        
        
        pre_norm = batch.clone()
        batch = self.pre_processors(batch, in_place=False)
        if not hasattr(self, "output_dir"):

            self.output_dir = "/home/ubuntu/bw-dl/rollout_outputs_" + str(uuid.uuid4())
        print(self.output_dir)
        

        with torch.no_grad():

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            # Dimensions are
            # batch, timesteps, horizonal space, variables
            x = batch[:, 0 : self.multi_step, None, ...]  # add dummy ensemble dimension as 3rd index
            output_dir = self.output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Determine next index using only existing y_hat_*.npy or y_hat_*.npy.npz files
            existing_indices = len(os.listdir(output_dir))
            new_index = existing_indices
            print(new_index)



            y_hat = self(x)            
            print(y_hat.shape)
            np.savez_compressed(
                f"{output_dir}/y_hat_{new_index}.npy",
                y_hat=y_hat.cpu().numpy(),
            )
            np.savez_compressed(
                f"{output_dir}/x_pre_norm_{new_index}.npy",
                x_pre_norm=pre_norm.cpu().numpy(),
            )
            # also save x
            np.savez_compressed(
                f"{output_dir}/x_{new_index}.npy",
                x=x.cpu().numpy(),
            )

            # # Save variable order for outputs (Y) and inputs (X) once per directory

            # # Build ordered output variable names from name->index mapping
            # y_map = self.data_indices.data.output.name_to_index
            # y_names = [None] * len(y_map)
            # for name, idx in y_map.items():
            #     y_names[idx] = str(name)

            # # Build ordered input variable names from name->index mapping
            # x_map = self.data_indices.data.input.name_to_index
            # x_names = [None] * len(x_map)
            # for name, idx in x_map.items():
            #     x_names[idx] = str(name)

            # # Write a variables.json for Y order (compatible with comparator script)
            # vars_json = os.path.join(output_dir, "variables.json")
            # if not os.path.exists(vars_json):
            #     with open(vars_json, "w") as f:
            #         json.dump({"variables": y_names}, f)

            # # Additionally write separate files for clarity
            # y_json = os.path.join(output_dir, "y_variables.json")
            # if not os.path.exists(y_json):
            #     with open(y_json, "w") as f:
            #         json.dump(y_names, f)

            # x_json = os.path.join(output_dir, "x_variables.json")
            # if not os.path.exists(x_json):
            #     with open(x_json, "w") as f:
            #         json.dump(x_names, f)

            # # Derive variables_in (variables minus diagnostic) and variables_out (variables minus forcing)
            # cfg = getattr(self.data_indices, "config", {})
            # data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
            # forcing_cfg = set(data_cfg.get("forcing", []) or [])
            # diagnostic_cfg = set(data_cfg.get("diagnostic", []) or [])

            # variables_in = [n for n in y_names if n not in diagnostic_cfg]
            # variables_out = [n for n in y_names if n not in forcing_cfg]

            # with open(os.path.join(output_dir, "variables_in.json"), "w") as f:
            #     json.dump(variables_in, f)
            # with open(os.path.join(output_dir, "variables_out.json"), "w") as f:
            #     json.dump(variables_out, f)

        return self.post_processors(y_hat, in_place=False)
