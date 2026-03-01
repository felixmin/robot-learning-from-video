# HLRP Containers

This repo maintains two container build targets:

- `containers/Dockerfile.stage12`
  Builds the stage-1/2 image for LAQ and foundation training.
  Docker tag: `felixmin/hlrp:stage12`
  Cluster image path: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage12.sqsh`
- `containers/Dockerfile.stage3`
  Builds the stage-3 LeRobot image.
  Docker tag: `felixmin/hlrp:stage3`
  Cluster image path: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/hlrp_stage3_lerobot.sqsh`

The preferred workflow is:

1. Build on the workstation.
2. Push the matching Docker tag to Docker Hub.
3. Import the tag to the cluster with Enroot.
4. Swap the imported `.sqsh` into the matching target path.

For stage 3, the image contains the baseline LeRobot runtime from the published package. During active development, jobs can overlay mounted source from this repo with:

```bash
pip install --no-deps --no-build-isolation -e /dss/.../high-level-robot-planner/lerobot
pip install --no-deps --no-build-isolation -e /dss/.../high-level-robot-planner/lerobot_policy_hlrp
```
