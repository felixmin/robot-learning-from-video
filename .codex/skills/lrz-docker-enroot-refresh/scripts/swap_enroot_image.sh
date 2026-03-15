#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

REPLACEMENT=""
CONFIG_PATH="${REPO_ROOT}/config/user_config/local.yaml"
ENROOT_ROOT="/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot"
ARCHIVE_DIR=""
DOCKER_TAG="unknown"
DIGEST="unknown"
IMPORT_JOB_ID="unknown"
ALLOW_ACTIVE_JOBS=0
BOOTSTRAP=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: swap_enroot_image.sh --replacement PATH [options]

Archive the previously configured unified LRZ image into enroot/old/ and
update a target config file to point at the new imported .sqsh.

Options:
  --replacement PATH           New imported .sqsh to activate.
  --config-path PATH           Config file to update. Default: config/user_config/local.yaml
  --enroot-root PATH           Expected DSS enroot root. Default: /dss/.../felix_minzenmay/enroot
  --archive-dir PATH           Archive directory. Default: <enroot-root>/old
  --docker-tag TAG             Docker tag for provenance. Default: unknown
  --digest SHA                 Pushed manifest digest for provenance. Default: unknown
  --import-job-id JOBID        Slurm import job id for provenance. Default: unknown
  --bootstrap                  Allow missing/nonstandard current image state and update config without archive.
  --allow-active-jobs          Skip the default queued/running Slurm job guard.
  --dry-run                    Print planned actions without executing them.
  -h, --help                   Show this message.
EOF
}

prepare_updated_config() {
  local src="$1"
  local dst="$2"
  local replacement="$3"

  python3 - "$src" "$dst" "$replacement" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
replacement = sys.argv[3]

if src.exists():
    lines = src.read_text().splitlines(keepends=True)
else:
    lines = ["# @package _global_\n", "\n"]

in_cluster = False
cluster_indent = None
in_container = False
container_indent = None
replaced = False

for idx, line in enumerate(lines):
    stripped = line.strip()
    indent = len(line) - len(line.lstrip(" "))

    if re.fullmatch(r"cluster:\s*", stripped):
        in_cluster = True
        cluster_indent = indent
        in_container = False
        continue

    if in_cluster and stripped and not stripped.startswith("#") and indent <= cluster_indent:
        in_cluster = False
        in_container = False

    if in_cluster and re.fullmatch(r"container:\s*", stripped) and indent > cluster_indent:
        in_container = True
        container_indent = indent
        continue

    if in_container and stripped and not stripped.startswith("#") and indent <= container_indent:
        in_container = False

    if in_container and re.match(r"image:\s*", stripped) and indent > container_indent:
        lines[idx] = "    image: " + replacement + "\n"
        replaced = True
        break

if not replaced:
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.extend([
        "cluster:\n",
        "  container:\n",
        f"    image: {replacement}\n",
    ])

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("".join(lines))
PY
}

read_current_image() {
  local config_path="$1"

  python3 - "$config_path" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("missing_config\t")
    raise SystemExit(0)

lines = path.read_text().splitlines()
in_cluster = False
cluster_indent = None
in_container = False
container_indent = None

for line in lines:
    stripped = line.strip()
    indent = len(line) - len(line.lstrip(" "))

    if re.fullmatch(r"cluster:\s*", stripped):
        in_cluster = True
        cluster_indent = indent
        in_container = False
        continue

    if in_cluster and stripped and not stripped.startswith("#") and indent <= cluster_indent:
        in_cluster = False
        in_container = False

    if in_cluster and re.fullmatch(r"container:\s*", stripped) and indent > cluster_indent:
        in_container = True
        container_indent = indent
        continue

    if in_container and stripped and not stripped.startswith("#") and indent <= container_indent:
        in_container = False

    if in_container and re.match(r"image:\s*", stripped) and indent > container_indent:
        value = stripped.split(":", 1)[1].strip()
        if value in {"", "null"}:
            print("missing_image\t")
        else:
            print(f"found\t{value}")
        raise SystemExit(0)

print("missing_image\t")
PY
}

remote_run() {
  local raw="$1"
  local remote_cmd

  printf -v remote_cmd 'bash -lc %q' "${raw}"
  printf '+ ssh ai %q\n' "${remote_cmd}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    ssh ai "${remote_cmd}"
  fi
}

remote_exists() {
  local path="$1"
  local raw
  local remote_cmd

  raw="test -e \"${path}\""
  printf -v remote_cmd 'bash -lc %q' "${raw}"
  ssh ai "${remote_cmd}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replacement)
      REPLACEMENT="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --enroot-root)
      ENROOT_ROOT="$2"
      shift 2
      ;;
    --archive-dir)
      ARCHIVE_DIR="$2"
      shift 2
      ;;
    --docker-tag)
      DOCKER_TAG="$2"
      shift 2
      ;;
    --digest)
      DIGEST="$2"
      shift 2
      ;;
    --import-job-id)
      IMPORT_JOB_ID="$2"
      shift 2
      ;;
    --bootstrap)
      BOOTSTRAP=1
      shift
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

if [[ -z "${REPLACEMENT}" ]]; then
  echo "--replacement is required" >&2
  exit 1
fi

[[ -n "${ARCHIVE_DIR}" ]] || ARCHIVE_DIR="${ENROOT_ROOT}/old"

case "${REPLACEMENT}" in
  "${ENROOT_ROOT}"/*.sqsh) ;;
  *)
    echo "Replacement must be an imported .sqsh under ${ENROOT_ROOT}: ${REPLACEMENT}" >&2
    exit 1
    ;;
esac

CONFIG_PATH=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${CONFIG_PATH}")
TIMESTAMP=$(date +%F_%H-%M-%S)
ACTIVATION_LOG="${ARCHIVE_DIR}/activation_log.tsv"

if [[ "${ALLOW_ACTIVE_JOBS}" -eq 0 ]]; then
  if ssh ai 'squeue --me -h | grep -q .'; then
    echo "Queued or running Slurm jobs detected. Rerun with --allow-active-jobs if you explicitly want to activate anyway." >&2
    exit 1
  fi
fi

if ! remote_exists "${REPLACEMENT}"; then
  echo "Replacement image not found on cluster: ${REPLACEMENT}" >&2
  exit 1
fi

IFS=$'\t' read -r CURRENT_STATUS CURRENT_IMAGE <<<"$(read_current_image "${CONFIG_PATH}")"

if [[ "${CURRENT_STATUS}" == "found" && "${CURRENT_IMAGE}" == "${REPLACEMENT}" ]]; then
  echo "Activation already current: ${REPLACEMENT}"
  exit 0
fi

CURRENT_INSIDE_ROOT=0
CURRENT_EXISTS_REMOTE=0
if [[ -n "${CURRENT_IMAGE:-}" ]]; then
  case "${CURRENT_IMAGE}" in
    "${ENROOT_ROOT}"/*)
      CURRENT_INSIDE_ROOT=1
      ;;
  esac
  if remote_exists "${CURRENT_IMAGE}"; then
    CURRENT_EXISTS_REMOTE=1
  fi
fi

if [[ "${CURRENT_STATUS}" == "missing_config" && "${BOOTSTRAP}" -eq 0 ]]; then
  echo "Config file not found: ${CONFIG_PATH}. Use --bootstrap to initialize it." >&2
  exit 1
fi

if [[ "${CURRENT_STATUS}" == "missing_image" && "${BOOTSTRAP}" -eq 0 ]]; then
  echo "No current cluster.container.image found in ${CONFIG_PATH}. Use --bootstrap to initialize it." >&2
  exit 1
fi

if [[ -n "${CURRENT_IMAGE:-}" && "${CURRENT_INSIDE_ROOT}" -eq 0 && "${BOOTSTRAP}" -eq 0 ]]; then
  echo "Current image is outside ${ENROOT_ROOT}: ${CURRENT_IMAGE}. Use --bootstrap to continue without archive." >&2
  exit 1
fi

if [[ -n "${CURRENT_IMAGE:-}" && "${CURRENT_INSIDE_ROOT}" -eq 1 && "${CURRENT_EXISTS_REMOTE}" -eq 0 && "${BOOTSTRAP}" -eq 0 ]]; then
  echo "Current image is configured under ${ENROOT_ROOT} but missing on cluster: ${CURRENT_IMAGE}. Use --bootstrap to continue without archive." >&2
  exit 1
fi

TMP_CONFIG=$(mktemp "${TMPDIR:-/tmp}/hlrp-local-config.XXXXXX")
trap 'rm -f "${TMP_CONFIG}"' EXIT
prepare_updated_config "${CONFIG_PATH}" "${TMP_CONFIG}" "${REPLACEMENT}"

ARCHIVED_PATH=""
if [[ -n "${CURRENT_IMAGE:-}" && "${CURRENT_INSIDE_ROOT}" -eq 1 && "${CURRENT_EXISTS_REMOTE}" -eq 1 ]]; then
  CURRENT_BASENAME=$(basename "${CURRENT_IMAGE}" .sqsh)
  ARCHIVED_PATH="${ARCHIVE_DIR}/${CURRENT_BASENAME}_${TIMESTAMP}.sqsh"
fi

printf 'Replacement: %s\n' "${REPLACEMENT}"
printf 'Config Path: %s\n' "${CONFIG_PATH}"
printf 'Enroot Root: %s\n' "${ENROOT_ROOT}"
printf 'Archive Dir: %s\n' "${ARCHIVE_DIR}"
printf 'Docker Tag: %s\n' "${DOCKER_TAG}"
printf 'Digest: %s\n' "${DIGEST}"
printf 'Import Job ID: %s\n' "${IMPORT_JOB_ID}"
printf 'Bootstrap: %s\n' "${BOOTSTRAP}"
if [[ -n "${CURRENT_IMAGE:-}" ]]; then
  printf 'Current Image: %s\n' "${CURRENT_IMAGE}"
else
  printf 'Current Image: <unset>\n'
fi
if [[ -n "${ARCHIVED_PATH}" ]]; then
  printf 'Archived Path: %s\n' "${ARCHIVED_PATH}"
else
  printf 'Archived Path: <none>\n'
fi

if [[ -n "${ARCHIVED_PATH}" ]]; then
  remote_run "set -euo pipefail; mkdir -p \"${ARCHIVE_DIR}\"; test ! -e \"${ARCHIVED_PATH}\"; mv \"${CURRENT_IMAGE}\" \"${ARCHIVED_PATH}\"; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' '${TIMESTAMP}' '${CURRENT_IMAGE}' '${ARCHIVED_PATH}' '${REPLACEMENT}' '${DOCKER_TAG}' '${DIGEST}' '${IMPORT_JOB_ID}' '${CONFIG_PATH}' >> \"${ACTIVATION_LOG}\"; ls -lh \"${ARCHIVED_PATH}\" \"${REPLACEMENT}\""
else
  remote_run "set -euo pipefail; mkdir -p \"${ARCHIVE_DIR}\"; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' '${TIMESTAMP}' '${CURRENT_IMAGE:-}' '' '${REPLACEMENT}' '${DOCKER_TAG}' '${DIGEST}' '${IMPORT_JOB_ID}' '${CONFIG_PATH}' >> \"${ACTIVATION_LOG}\"; ls -lh \"${REPLACEMENT}\""
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mv "${TMP_CONFIG}" "${CONFIG_PATH}"
  trap - EXIT
  printf 'Updated %s\n' "${CONFIG_PATH}"
else
  printf 'Config update preview:\n'
  sed -n '1,120p' "${TMP_CONFIG}"
fi
