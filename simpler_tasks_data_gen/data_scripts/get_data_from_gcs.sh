#!/bin/bash 
gsutil -m cp -r gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/$1/**/train* /mnt/disks/persist/Universal_LIME/data/
gsutil -m cp -r gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/$1/**/valid* /mnt/disks/persist/Universal_LIME/data/


# gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/1-11_t5_token_basic_unary_tasks