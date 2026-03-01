#!/usr/bin/env bash
set -euo pipefail

PROFILE=""
TARGET=""
REPLACEMENT=""
BACKUP=""
ALLOW_ACTIVE_JOBS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: swap_enroot_image.sh --profile PROFILE --replacement PATH [options]

Rename the active stage-specific image to a dated backup and move the replacement image into place.

Options:
  --profile PROFILE            One of: stage12, stage3.
  --replacement PATH           New .sqsh file to move into place.
  --target PATH                Target image path. Defaults from --profile.
  --backup PATH                Explicit backup path. Defaults to target with a timestamp suffix.
  --allow-active-jobs          Skip the default squeue safety check.
  --dry-run                    Print the ssh command without executing it.
  -h, --help                   Show this message.
EOF
}

apply_profile_defaults() {
  case "${PROFILE}" in
    stage12|default|laq|foundation)
      PROFILE="stage12"
      [[ -n "${TARGET}" ]] || TARGET="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage12.sqsh"
      ;;
    stage3|lerobot|libero)
      PROFILE="stage3"
      [[ -n "${TARGET}" ]] || TARGET="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_lerobot.sqsh"
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
    --replacement)
      REPLACEMENT="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --backup)
      BACKUP="$2"
      shift 2
      ;;
    --allow-active-jobs)
      ALLOW_ACTIVE_JOBS=1
      shift
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

if [[ -z "${REPLACEMENT}" ]]; then
  echo "--replacement is required" >&2
  exit 1
fi

if [[ -z "${BACKUP}" ]]; then
  TARGET_DIR=$(dirname "${TARGET}")
  TARGET_BASENAME=$(basename "${TARGET}" .sqsh)
  BACKUP="${TARGET_DIR}/${TARGET_BASENAME}_$(date +%F_%H-%M-%S)_pre_refresh.sqsh"
fi

if [[ "${REPLACEMENT}" == "${TARGET}" ]]; then
  echo "Replacement and target must differ" >&2
  exit 1
fi

if [[ "${ALLOW_ACTIVE_JOBS}" -eq 0 ]]; then
  if ssh ai 'squeue --me -h | grep -q .'; then
    echo "Active Slurm jobs detected. Rerun with --allow-active-jobs if the user explicitly wants to swap anyway." >&2
    exit 1
  fi
fi

REMOTE_RAW="set -euo pipefail; test -e \"${REPLACEMENT}\"; test -e \"${TARGET}\"; test ! -e \"${BACKUP}\"; mv \"${TARGET}\" \"${BACKUP}\"; mv \"${REPLACEMENT}\" \"${TARGET}\"; ls -lh \"${TARGET}\" \"${BACKUP}\""
printf -v REMOTE_CMD '%q' "${REMOTE_RAW}"

printf 'Profile: %s\n' "${PROFILE}"
printf 'Target: %s\n' "${TARGET}"
printf 'Replacement: %s\n' "${REPLACEMENT}"
printf '+ ssh ai %q\n' "${REMOTE_CMD}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  ssh ai "${REMOTE_CMD}"
fi
