#!/bin/bash 

gsutil rm -rf gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/$1/
gsutil -m cp -r /mnt/disks/persist/Universal_LIME/data/$1/ gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/
# gsutil -m cp -r /mnt/disks/persist/Universal_LIME/data/$1/ gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/



# gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/1-11_t5_token_basic_unary_tasks