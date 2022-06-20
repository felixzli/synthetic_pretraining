
parent_folder=$(basename $(dirname "$0"))
gsutil -m cp -r gs://n2formal-community-external-lime/universal_lime/synthetic_tasks_data/$parent_folder /mnt/disks/persist/felix-text-to-text-transfer-transformer/synthetic_tasks_data/
