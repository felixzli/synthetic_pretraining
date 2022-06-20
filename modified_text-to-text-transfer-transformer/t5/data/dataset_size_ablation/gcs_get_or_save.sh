if [ "$1" = "g" ]
then
#!/bin/bash

gsutil -m cp -r gs://n2formal-community-external-lime/universal_lime/dataset_size_ablation/data/ \
t5/data/dataset_size_ablation
fi


if [ "$1" = "s" ]
then
echo save

gsutil -m cp -r \
t5/data/dataset_size_ablation/data/* gs://n2formal-community-external-lime/universal_lime/dataset_size_ablation/data/

gsutil ls gs://n2formal-community-external-lime/universal_lime/dataset_size_ablation/data/
fi
