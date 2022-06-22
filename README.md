
# Overview
This repo contains scripts to reproduce experiments in the paper: 
[Insights into Pre-training via Simpler Synthetic Tasks](https://arxiv.org/abs/2206.10139)


# Setup
Use a Google Cloud v3 or v4 TPU VM.
Run the below commands:
```
cd ./modified_text-to-text-transfer-transformer
python3 -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd ..
cd modified_text-to-text-transfer-transformer
pip install -e .
cd ..
pip uninstall jax jaxlib libtpu-nightly libtpu libtpu_tpuv3 libtpu_tpuv4 -y
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```



# Data
Before pre-training or fine-tuning, you must first download all pre-training and fine-tuning data. Run:
```
bash run_scripts_to_reproduce_experiments/data_scripts/get_all_pretraining_and_finetuning_data_from_gcs.sh
```
The above command will save data to `data/data_from_gcs`.



To generate your own 18 simpler pre-training task data run:
```
bash run_scripts_to_reproduce_experiments/data_scripts/generate_simpler_tasks_data.sh
```

# Training

## Before running any training scripts, do the following:

**You must `cd` to modified_text-to-text-transfer-transformer. Run all training scripts from inside there. For example you might run the command `bash ../run_scripts_to_reproduce_experiments/training_scripts/finetuning/cnndm_10k/table1_exp.sh`. Note the `..` in the command.**

**Also, you must set the variable `DATA_BASE_DIR` in the file `modified_text-to-text-transfer-transformer/t5/data/mar31_and_after_tasks/add_tasks_utils.py`  to the absolute path of your downloaded data from above. For example, to run the code on my VMs I set `DATA_BASE_DIR = '/mnt/disks/persist/draft_synthetic_pretraining_code/data/data_from_gcs/'`**


## Pre-training
First, edit `run_scripts_to_reproduce_experiments/training_scripts/pretraining/pretrain.sh` two arguments:
```
exp_dir
task_id
``` 
`task_id` is the task you pre-train on. It can be set to: lime, nonsense_summary, nesting_language, set, identity, delete, sort, union, replace, duplicate, intersect, reverse, deduplicate, search,longest_word, length, count, first_token, last_token, set1_minus_set2, set2_minus_set1

`exp_dir` is where the checkpoints and results of your training will be saved.


## Fine-tuning
### Table 1 Results

Edit variables in the file `run_scripts_to_reproduce_experiments/training_scripts/finetuning/*/table1_exp.sh` and then run it. The `*` can be cnndm_10k, code_translation, mtop, retrosynthesis, squad, or webqsp.
The variables to be editted are `exp_dir` (same definition as above) and `ckpt`, which is either the absolute path to your pre-trained model checkpoint that you want to load, or the text `from_scratch` which will not load any checkpoint and reproduce the results labelled  `Random Init` in the paper. For CNNDM-10K, as mentioned in the paper, T5v1.1-small instead of T5-small was used to evaluate off-the-shelf language pre-training. To run this experiment, instead of `table1_exp.sh` you must run `run_scripts_to_reproduce_experiments/training_scripts/finetuning/cnndm_10k/table1_t511_exp.sh`.


### Table 2 Init Results
Edit variables in the file `run_scripts_to_reproduce_experiments/training_scripts/finetuning/*/table2_init_exp.sh` and then run it. The `*` can be `cnndm_10k`, `code_translation`, `mtop`, `retrosynthesis`, `squad`, or `webqsp`.
The variables to be edited are `exp_dir` (same definition as above),`ckpt` (same definition as above), and `init_id`, which specifies what initialization scheme you want to use. `init_id` can be:
```
per_param_grouping___init_mean_std (Per Param Mean/SD in paper)
per_param_grouping___init_scale (Per Param Scale in paper)
across_ALL_layer_AND_per_layer_grouping___init_scale (Whole Model Scale in paper)
preattnln_per_param_grouping___init_scale (Pre-attn LN Per Param Scale in paper)
```
For the additional initialization schemes tried in the appendix, `init_id` can be:
```
across_ALL_layer_grouping___init_scale (Across Layers Scale in paper)
per_layer_grouping___init_scale (Per Layer Scale in paper)
premlpln_per_param_grouping___init_scale (Pre-MLP LN Per Param Scale in paper)
qkvo_per_param_grouping___init_scale (Attention Per Param Scale in paper)
mlp_per_param_grouping___init_scale (MLP Per Param Scale in paper)
```


### Table 3 Init Pre-attention Layer Norm Results
Edit variables in the file `run_scripts_to_reproduce_experiments/training_scripts/finetuning/*/table3_pre_attn_ln_exp.sh` and then run it. The `*` can be `cnndm_10k`, `code_translation`, `mtop`, `retrosynthesis`, `squad`, or `webqsp`.
The variables to be edited are `exp_dir` (same definition as above) and `init_pre_attn_ln_value`, which is the value you want to initialize the pre-attention layer norms to.
