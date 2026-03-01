#!/usr/bin/env bash
set -euo pipefail

PROFILE=""
IMAGE_URI=""
PARTITION="lrz-cpu"
QOS="cpu"
TIME_LIMIT="01:00:00"
MEMORY="128G"
CPUS="4"
MAX_PROCESSORS="4"
JOB_NAME=""
OUTPUT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_enroot_import.sh [options]

Submit the validated high-memory Enroot import job on ssh ai.

Options:
  --profile PROFILE            One of: stage12, stage3.
  --image-uri URI              Enroot image URI. Defaults from --profile.
  --output PATH                Cluster output .sqsh path. Defaults from --profile.
  --partition PARTITION        Slurm partition. Default: lrz-cpu
  --qos QOS                    Slurm qos. Default: cpu
  --time LIMIT                 Slurm time limit. Default: 01:00:00
  --mem MEMORY                 Slurm memory request. Default: 128G
  --cpus N                     Slurm cpus-per-task. Default: 4
  --max-processors N           ENROOT_MAX_PROCESSORS. Default: 4.
  --job-name NAME              Slurm job name. Defaults from --profile.
  --dry-run                    Print the ssh/sbatch command without executing it.
  -h, --help                   Show this message.
EOF
}

apply_profile_defaults() {
  case "${PROFILE}" in
    stage12|default|laq|foundation)
      PROFILE="stage12"
      [[ -n "${IMAGE_URI}" ]] || IMAGE_URI="docker://felixmin/hlrp:stage12"
      [[ -n "${JOB_NAME}" ]] || JOB_NAME="enroot-import-hlrp-stage12"
      [[ -n "${OUTPUT}" ]] || OUTPUT="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage12_$(date +%F_%H-%M-%S).sqsh"
      ;;
    stage3|lerobot|libero)
      PROFILE="stage3"
      [[ -n "${IMAGE_URI}" ]] || IMAGE_URI="docker://felixmin/hlrp:stage3"
      [[ -n "${JOB_NAME}" ]] || JOB_NAME="enroot-import-hlrp-stage3"
      [[ -n "${OUTPUT}" ]] || OUTPUT="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_$(date +%F_%H-%M-%S).sqsh"
      ;;
    *)
      echo "Unsupported --profile: ${PROFILE}" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --image-uri)
      IMAGE_URI="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --max-processors)
      MAX_PROCESSORS="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PROFILE}" ]]; then
  echo "--profile is required (stage12 or stage3)" >&2
  usage >&2
  exit 1
fi

apply_profile_defaults

if [[ "${IMAGE_URI}" == docker://docker.io/* ]]; then
  echo "Use docker://<namespace>/<image>:<tag> for Docker Hub, not docker://docker.io/..." >&2
  exit 1
fi

OUTPUT_DIR=$(dirname "${OUTPUT}")
WRAP_RAW="set -euo pipefail; test ! -e \"${OUTPUT}\"; mkdir -p \"${OUTPUT_DIR}\"; export ENROOT_MAX_PROCESSORS=\"${MAX_PROCESSORS}\"; enroot import -o \"${OUTPUT}\" \"${IMAGE_URI}\"; ls -lh \"${OUTPUT}\""
printf -v REMOTE_CMD 'sbatch -p %q -q %q -t %q --mem=%q -c %q -J %q --wrap %q' \
  "${PARTITION}" "${QOS}" "${TIME_LIMIT}" "${MEMORY}" "${CPUS}" "${JOB_NAME}" "${WRAP_RAW}"

printf 'Profile: %s\n' "${PROFILE}"
printf 'Image URI: %s\n' "${IMAGE_URI}"
printf 'Output: %s\n' "${OUTPUT}"
printf '+ ssh ai %q\n' "${REMOTE_CMD}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  ssh ai "${REMOTE_CMD}"
fi
