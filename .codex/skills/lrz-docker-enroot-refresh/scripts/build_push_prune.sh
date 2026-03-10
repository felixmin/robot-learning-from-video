#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

PROFILE=""
IMAGE=""
DOCKERFILE=""
CONTEXT=""
PRUNE_BEFORE=1
PRUNE_AFTER=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: build_push_prune.sh [options]

Build and push a stage-specific HLRP image from the workstation, then leave Docker empty.

Options:
  --profile PROFILE             One of: stage12, stage3.
  --image IMAGE                 Docker tag to build and push.
  --dockerfile PATH             Path to Dockerfile. Defaults from --profile.
  --context PATH                Docker build context. Defaults from --profile.
  --skip-prune-before           Skip the initial Docker prune.
  --skip-prune-after            Skip the final Docker prune.
  --dry-run                     Print commands without executing them.
  -h, --help                    Show this message.
EOF
}

run() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

apply_profile_defaults() {
  case "${PROFILE}" in
    stage12|default|laq|foundation)
      PROFILE="stage12"
      [[ -n "${IMAGE}" ]] || IMAGE="felixmin/hlrp:stage12"
      [[ -n "${DOCKERFILE}" ]] || DOCKERFILE="${REPO_ROOT}/containers/Dockerfile.stage12"
      [[ -n "${CONTEXT}" ]] || CONTEXT="${REPO_ROOT}"
      ;;
    stage3|lerobot|libero)
      PROFILE="stage3"
      [[ -n "${IMAGE}" ]] || IMAGE="felixmin/hlrp:stage3"
      [[ -n "${DOCKERFILE}" ]] || DOCKERFILE="${REPO_ROOT}/containers/Dockerfile.stage3"
      [[ -n "${CONTEXT}" ]] || CONTEXT="${REPO_ROOT}"
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
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE="$2"
      shift 2
      ;;
    --context)
      CONTEXT="$2"
      shift 2
      ;;
    --skip-prune-before)
      PRUNE_BEFORE=0
      shift
      ;;
    --skip-prune-after)
      PRUNE_AFTER=0
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

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

case "${PROFILE}" in
  stage12)
    if [[ ! -f "${CONTEXT}/requirements.txt" ]]; then
      echo "Expected build context to contain requirements.txt: ${CONTEXT}" >&2
      exit 1
    fi
    if [[ ! -d "${CONTEXT}/packages" ]]; then
      echo "Expected build context to contain packages/: ${CONTEXT}" >&2
      exit 1
    fi
    ;;
esac

printf 'Profile: %s\n' "${PROFILE}"
printf 'Image: %s\n' "${IMAGE}"
printf 'Dockerfile: %s\n' "${DOCKERFILE}"
printf 'Context: %s\n' "${CONTEXT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '\n'
fi

if [[ "${PRUNE_BEFORE}" -eq 1 ]]; then
  run docker system prune -a --volumes -f
  run docker builder prune -a -f
fi

run docker build -f "${DOCKERFILE}" -t "${IMAGE}" "${CONTEXT}"
run docker push "${IMAGE}"

if [[ "${PRUNE_AFTER}" -eq 1 ]]; then
  run docker system prune -a --volumes -f
  run docker builder prune -a -f
fi

run docker system df
