"""
HPO Quickstart with PyTorch
===========================
This tutorial optimizes the model in `official PyTorch quickstart`_ with auto-tuning.

The tutorial consists of 4 steps:

1. Modify the model for auto-tuning.
2. Define hyperparameters' search space.
3. Configure the experiment.
4. Run the experiment.

.. _official PyTorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

# %%
# Step 1: Prepare the model
# -------------------------
# In first step, we need to prepare the model to be tuned.
#
# The model should be put in a separate script.
# It will be evaluated many times concurrently,
# and possibly will be trained on distributed platforms.
#
# In this tutorial, the model is defined in :doc:`model.py <model>`.
#
# In short, it is a PyTorch model with 3 additional API calls:
#
# 1. Use :func:`nni.get_next_parameter` to fetch the hyperparameters to be evalutated.
# 2. Use :func:`nni.report_intermediate_result` to report per-epoch accuracy metrics.
# 3. Use :func:`nni.report_final_result` to report final accuracy.
#
# Please understand the model code before continue to next step.

# %%
# Step 2: Define search space
# ---------------------------
# In model code, we have prepared 3 hyperparameters to be tuned:
# *features*, *lr*, and *momentum*.
#
# Here we need to define their *search space* so the tuning algorithm can sample them in desired range.
#
# Assuming we have following prior knowledge for these hyperparameters:
#
# 1. *features* should be one of 128, 256, 512, 1024.
# 2. *lr* should be a float between 0.0001 and 0.1, and it follows exponential distribution.
# 3. *momentum* should be a float between 0 and 1.
#
# In NNI, the space of *features* is called ``choice``;
# the space of *lr* is called ``loguniform``;
# and the space of *momentum* is called ``uniform``.
# You may have noticed, these names are derived from ``numpy.random``.
#
# For full specification of search space, check :doc:`the reference </hpo/search_space>`.
#
# Now we can define the search space as follow:

search_space = {
    'iou_thr': {'_type': 'quniform', '_value': [0.05, 0.9, 0.05]},
    'skip_iou_thr': {'_type': 'quniform', '_value': [0.01, 0.2, 0.02]},
    'w0': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    'w1': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    'w2': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w3': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w4': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w5': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w6': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w7': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w8': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w9': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w10': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w11': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w12': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w13': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w14': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w15': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w16': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w17': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w18': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w19': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w20': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w21': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w22': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w23': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w24': {'_type': 'quniform', '_value': [0, 1, 0.1]},
    #'w25': {'_type': 'quniform', '_value': [0, 1, 0.1]},
}

# %%
# Step 3: Configure the experiment
# --------------------------------
# NNI uses an *experiment* to manage the HPO process.
# The *experiment config* defines how to train the models and how to explore the search space.
#
# In this tutorial we use a *local* mode experiment,
# which means models will be trained on local machine, without using any special training platform.
from nni.experiment import Experiment
experiment = Experiment('local')

experiment.config.trial_command = 'python ./run.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space

experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.tuner.class_args['population_size'] = 100
experiment.config.assessor.name = 'Medianstop'

experiment.config.max_trial_number = 400
experiment.config.trial_concurrency = 8
#GPU
experiment.config.trial_gpu_number = 1
experiment.config.training_service.use_active_gpu = False


experiment.run(23789)
experiment.stop()
