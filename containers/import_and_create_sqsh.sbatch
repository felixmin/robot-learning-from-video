#!/bin/bash
#SBATCH -p lrz-v100x2,lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-dgx-a100-40x8-mig,lrz-dgx-1-v100x8,lrz-dgx-1-p100x8,lrz-hpe-p100x4
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -J enroot-import

set -euo pipefail

IMG="$HOME/containers/hlrp_stage12.sqsh"
REMOTE="docker://felixmin/hlrp:stage12"

mkdir -p "$(dirname "$IMG")"

echo "Importing to: $IMG"
enroot import -o "$IMG" "$REMOTE"

echo "Done. Image stored at: $IMG"
ls -lh "$IMG"
