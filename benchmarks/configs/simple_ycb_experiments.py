# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import SimpleYcbExperiments
from benchmarks.configs.ycb_experiments import experiments
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
)

# Adding Resampling to YCB experiments
simple_ycb_experiments = {}
for exp_name, cfg in asdict(experiments).items():
    if exp_name in [
        "base_config_10distinctobj_dist_agent",
        "randrot_noise_10distinctobj_dist_agent",
    ]:
        mod_exp_name = "simple_" + exp_name
        mod_cfg = cfg.copy()

        mod_cfg["eval_dataloader_args"]["object_names"] = ["mug"]
        mod_cfg["experiment_args"]["n_eval_epochs"] = 1
        mod_cfg["eval_dataloader_args"]["object_init_sampler"].rotations = [
            np.array([0, 0, 0])
        ]
        mod_cfg["logging_config"] = DetailedEvidenceLMLoggingConfig(
            monty_handlers=[BasicCSVStatsHandler, DetailedJSONHandler],
            wandb_handlers=[],
        )

        simple_ycb_experiments[mod_exp_name] = mod_cfg


experiments = SimpleYcbExperiments(
    simple_randrot_noise_10distinctobj_dist_agent=simple_ycb_experiments[
        "simple_randrot_noise_10distinctobj_dist_agent"
    ],
    simple_base_config_10distinctobj_dist_agent=simple_ycb_experiments[
        "simple_base_config_10distinctobj_dist_agent"
    ],
)


CONFIGS = asdict(experiments)
