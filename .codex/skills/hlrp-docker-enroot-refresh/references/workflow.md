# HLRP Docker Enroot Refresh Workflow

## Defaults

- Local repo root: `/mnt/data/workspace/code/high-level-robot-planner`
- Docker build context: `/mnt/data/workspace/code/high-level-robot-planner`
- Cluster host: `ai`
- Cluster Enroot dir: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot`

### Stage 1/2 Profile

- Dockerfile: `/mnt/data/workspace/code/high-level-robot-planner/containers/Dockerfile.stage12`
- Pushed tag: `felixmin/hlrp:stage12`
- Active cluster image path: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage12.sqsh`

### Stage 3 Profile

- Dockerfile: `/mnt/data/workspace/code/high-level-robot-planner/containers/Dockerfile.stage3`
- Pushed tag: `felixmin/hlrp:stage3`
- Active cluster image path: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_lerobot.sqsh`

## Validated Command Shapes

Local build/push:

```bash
docker system prune -a --volumes -f
docker builder prune -a -f
docker build \
  -f /mnt/data/workspace/code/high-level-robot-planner/containers/Dockerfile.stage3 \
  -t felixmin/hlrp:stage3 \
  /mnt/data/workspace/code/high-level-robot-planner
docker push felixmin/hlrp:stage3
docker system prune -a --volumes -f
docker builder prune -a -f
```

Cluster import:

```bash
sbatch -p lrz-cpu -q cpu -t 01:00:00 --mem=128G -c 4 -J enroot-import-hlrp-oli \
  --wrap "mkdir -p /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot && \
  export ENROOT_MAX_PROCESSORS=4 && \
  enroot import -o /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_<new-name>.sqsh \
  docker://felixmin/hlrp:stage3 && \
  ls -lh /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_<new-name>.sqsh"
```

Safe swap:

```bash
mv hlrp_stage3_lerobot.sqsh hlrp_stage3_lerobot_YYYY-MM-DD_pre_refresh.sqsh
mv <new-name>.sqsh hlrp_stage3_lerobot.sqsh
```

## Known Failure Modes

- `Dockerfile.stage3` expects the bundled `lerobot/` folder to be present in the repo root build context.
- `docker://docker.io/felixmin/hlrp:stage12` or `docker://docker.io/felixmin/hlrp:stage3` are the wrong Enroot URIs. Use `docker://felixmin/hlrp:stage12` or `docker://felixmin/hlrp:stage3`.
- `enroot import` can OOM while creating squashfs. The validated fix was `--mem=128G`, `-c 4`, and `ENROOT_MAX_PROCESSORS=4`.
- If an import job OOMs, the partial `.sqsh` may exist but should be treated as invalid. Do not delete it; create a new output filename for the retry.
- Do not replace `hlrp_stage12.sqsh` or `hlrp_stage3_lerobot.sqsh` while jobs are active unless the user explicitly accepts that risk.
