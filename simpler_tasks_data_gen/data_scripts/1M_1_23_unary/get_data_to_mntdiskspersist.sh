
parent_folder=$(basename $(dirname "$0"))
mkdir -p /mnt/disks/persist/synthetic_tasks_data/
gsutil -m cp -r gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/$parent_folder /mnt/disks/persist/synthetic_tasks_data/






# gsutil -m cp -r gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/1M_1_23_unary/length/ /mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/1M_1_23_unary/



