#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# Modify these arguments if you want to try other datasets, splits, or methods
dataset='inria_buildings'
split='0.5'
method='heightmatch'
model='dinov2_small'

config="configs/${dataset}.yaml"
save_path="saved/${dataset}/${split}/${method}/${model}"
labeled_id_path="splits/${dataset}/${split}/labeled.txt"
unlabeled_id_path="splits/${dataset}/${split}/unlabeled.txt"
val_id_path="splits/${dataset}/val.txt"

mkdir -p "$save_path"

# Automatically generate a random master port
master_port=$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)

echo "Using master port: $master_port"

torchrun --nproc_per_node="$1" \
         --master_addr=localhost \
         --master_port="$master_port" \
         "${method}.py" \
         --config="$config" \
         --labeled-id-path "$labeled_id_path" \
         --unlabeled-id-path "$unlabeled_id_path" \
         --val-id-path "$val_id_path" \
         --save-path "$save_path" 2>&1 | tee "$save_path/$now.log"