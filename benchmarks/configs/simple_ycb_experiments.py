# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict

from benchmarks.configs.names import SimpleYcbExperiments
from benchmarks.configs.ycb_experiments import experiments
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
)

simple_ycb_experiments = {}

for exp_name, cfg in asdict(experiments).items():
    if exp_name in [
        "base_config_10distinctobj_dist_agent",
        "randrot_noise_10distinctobj_dist_agent",
    ]:
        mod_exp_name = "simple_" + exp_name
        mod_cfg = cfg.copy()

        test_rotation = get_cube_face_and_corner_views_rotations()[2:3]
        mod_cfg["experiment_args"]["n_eval_epochs"] = len(test_rotation)
        mod_cfg["eval_dataloader_args"] = EnvironmentDataloaderPerObjectArgs(
            object_names=["mug", "banana", "potted_meat_can"],
            object_init_sampler=PredefinedObjectInitializer(rotations=test_rotation),
        )
        mod_cfg["logging_config"] = DetailedEvidenceLMLoggingConfig(
            monty_handlers=[
                BasicCSVStatsHandler,
                DetailedJSONHandler,
                # ReproduceEpisodeHandler,
            ],
            wandb_handlers=[],
        )

        # === MODS === #
        # updater_args = {
        #     "resampling_multiplier": 0.0,
        #     "evidence_slope_threshold": -1.0,
        #     "include_telemetry": True,
        # }

        # mod_cfg["monty_config"]["learning_module_configs"]["learning_module_0"][
        #     "learning_module_args"
        # ]["hypotheses_updater_class"] = ResamplingHypothesesUpdater

        # mod_cfg["monty_config"]["learning_module_configs"]["learning_module_0"][
        #     "learning_module_args"
        # ]["hypotheses_updater_args"].update(updater_args)
        # === END MODS === #

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
