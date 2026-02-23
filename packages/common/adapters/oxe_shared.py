"""Shared OpenX dataset registry + small access helpers."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataset config & registry
# ---------------------------------------------------------------------------


@dataclass
class OXEDatasetConfig:
    """Configuration for an OXE dataset."""

    name: str
    gcs_path: str
    image_key: str = "rgb"
    instruction_key: str = "instruction"
    state_key: Optional[str] = "effector_translation"
    image_shape: Tuple[int, int, int] = (360, 640, 3)
    control_frequency_hz: float = 10.0
    action_dim: int = 2
    state_dim: int = 2
    action_key: Optional[str] = None
    action_is_dict: bool = False
    instruction_in_step: bool = False
    robot_key: Optional[str] = None
    avg_episode_length: int = 30
    allow_missing_state: bool = False


OXE_DATASETS = {
    "language_table": OXEDatasetConfig(
        name="language_table",
        gcs_path="gs://gresearch/robotics/language_table/0.1.0",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "language_table_blocktorelative_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktorelative_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "language_table_blocktoblock_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoblock_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "language_table_blocktoabsolute_oracle_sim": OXEDatasetConfig(
        name="language_table_blocktoabsolute_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "language_table_separate_oracle_sim": OXEDatasetConfig(
        name="language_table_separate_oracle_sim",
        gcs_path="gs://gresearch/robotics/language_table_separate_oracle_sim/0.0.1",
        image_key="rgb",
        instruction_key="instruction",
        state_key="effector_translation",
        image_shape=(360, 640, 3),
        control_frequency_hz=10.0,
        avg_episode_length=40,
    ),
    "bridge": OXEDatasetConfig(
        name="bridge",
        gcs_path="gs://gresearch/robotics/bridge/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        state_key="state",
        image_shape=(480, 640, 3),
        control_frequency_hz=5.0,
        action_dim=3,
        state_dim=2,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=50,
    ),
    "rt1": OXEDatasetConfig(
        name="rt1",
        gcs_path="gs://gresearch/robotics/fractal20220817_data/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        state_key="base_pose_tool_reached",
        image_shape=(256, 320, 3),
        control_frequency_hz=3.0,
        action_dim=3,
        state_dim=3,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=30,
    ),
    "robonet": OXEDatasetConfig(
        name="robonet",
        gcs_path="gs://gresearch/robotics/robo_net/0.1.0",
        image_key="image",
        instruction_key="language_instruction",
        state_key="state",
        image_shape=(240, 320, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=3,
        action_key=None,
        action_is_dict=False,
        instruction_in_step=True,
        robot_key="robot",
        avg_episode_length=30,
    ),
    "aloha_mobile": OXEDatasetConfig(
        name="aloha_mobile",
        gcs_path="gs://gresearch/robotics/aloha_mobile/0.0.1",
        image_key="cam_high",
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,
        action_dim=16,
        state_dim=14,
        action_is_dict=False,
        avg_episode_length=200,
    ),
    "droid": OXEDatasetConfig(
        name="droid",
        gcs_path="gs://gresearch/robotics/droid/1.0.1",
        image_key="exterior_image_1_left",
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="cartesian_position",
        image_shape=(180, 320, 3),
        control_frequency_hz=10.0,
        action_dim=7,
        state_dim=6,
        action_is_dict=False,
        avg_episode_length=300,
    ),
    "berkeley_autolab_ur5": OXEDatasetConfig(
        name="berkeley_autolab_ur5",
        gcs_path="gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_state",
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=15,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=90,
    ),
    "jaco_play": OXEDatasetConfig(
        name="jaco_play",
        gcs_path="gs://gresearch/robotics/jaco_play/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="end_effector_cartesian_pos",
        image_shape=(224, 224, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=7,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=70,
    ),
    "kuka": OXEDatasetConfig(
        name="kuka",
        gcs_path="gs://gresearch/robotics/kuka/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key=None,
        image_shape=(512, 640, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=0,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=15,
        allow_missing_state=True,
    ),
    "taco_play": OXEDatasetConfig(
        name="taco_play",
        gcs_path="gs://gresearch/robotics/taco_play/0.1.0",
        image_key="rgb_static",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_obs",
        image_shape=(150, 200, 3),
        control_frequency_hz=10.0,
        action_dim=7,
        state_dim=15,
        action_key="actions",
        action_is_dict=True,
        avg_episode_length=60,
    ),
    "roboturk": OXEDatasetConfig(
        name="roboturk",
        gcs_path="gs://gresearch/robotics/roboturk/0.1.0",
        image_key="front_rgb",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key=None,
        image_shape=(480, 640, 3),
        control_frequency_hz=10.0,
        action_dim=3,
        state_dim=0,
        action_key="world_vector",
        action_is_dict=True,
        avg_episode_length=90,
        allow_missing_state=True,
    ),
    "bc_z": OXEDatasetConfig(
        name="bc_z",
        gcs_path="gs://gresearch/robotics/bc_z/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="present/xyz",
        state_dim=3,
        image_shape=(171, 213, 3),
        action_key="future/xyz_residual",
        action_is_dict=True,
        action_dim=30,
        avg_episode_length=30,
    ),
    "berkeley_cable_routing": OXEDatasetConfig(
        name="berkeley_cable_routing",
        gcs_path="gs://gresearch/robotics/berkeley_cable_routing/0.1.0",
        image_key="image",
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_state",
        state_dim=2,
        image_shape=(128, 128, 3),
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=30,
    ),
    "columbia_cairlab_pusht_real": OXEDatasetConfig(
        name="columbia_cairlab_pusht_real",
        gcs_path="gs://gresearch/robotics/columbia_cairlab_pusht_real/0.1.0",
        image_key="image",
        image_shape=(240, 320, 3),
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="robot_state",
        state_dim=2,
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=30,
    ),
    "mimic_play": OXEDatasetConfig(
        name="mimic_play",
        gcs_path="gs://gresearch/robotics/mimic_play/0.0.1",
        image_key="image/front_image_1",
        image_shape=(120, 120, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state/ee_pose",
        state_dim=7,
        action_dim=7,
        avg_episode_length=200,
    ),
    "berkeley_fanuc_manipulation": OXEDatasetConfig(
        name="berkeley_fanuc_manipulation",
        gcs_path="gs://gresearch/robotics/berkeley_fanuc_manipulation/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=6,
        avg_episode_length=60,
    ),
    "dobbe": OXEDatasetConfig(
        name="dobbe",
        gcs_path="gs://gresearch/robotics/dobbe/0.0.1",
        image_key="wrist_image",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "uiuc_d3field": OXEDatasetConfig(
        name="uiuc_d3field",
        gcs_path="gs://gresearch/robotics/uiuc_d3field/0.1.0",
        image_key="image_1",
        image_shape=(360, 640, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=3,
        avg_episode_length=60,
    ),
    "ucsd_kitchen_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="ucsd_kitchen_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(480, 640, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=8,
        avg_episode_length=60,
    ),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/ucsd_pick_and_place_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=4,
        avg_episode_length=60,
    ),
    "furniture_bench_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="furniture_bench_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/furniture_bench_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(224, 224, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=8,
        avg_episode_length=60,
    ),
    "maniskill_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="maniskill_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "robo_set": OXEDatasetConfig(
        name="robo_set",
        gcs_path="gs://gresearch/robotics/robo_set/0.0.1",
        image_key="image_left",
        image_shape=(240, 424, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=8,
        avg_episode_length=60,
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        name="stanford_hydra_dataset_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
        image_key="image",
        image_shape=(240, 320, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "stanford_robocook_converted_externally_to_rlds": OXEDatasetConfig(
        name="stanford_robocook_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/stanford_robocook_converted_externally_to_rlds/0.1.0",
        image_key="image_1",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "spoc": OXEDatasetConfig(
        name="spoc",
        gcs_path="gs://gresearch/robotics/spoc/0.0.1",
        image_key="image",
        image_shape=(224, 384, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        action_dim=9,
        avg_episode_length=60,
    ),
    "tidybot": OXEDatasetConfig(
        name="tidybot",
        gcs_path="gs://gresearch/robotics/tidybot/0.0.1",
        image_key="image",
        image_shape=(360, 640, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        action_dim=0,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        avg_episode_length=60,
    ),
    "toto": OXEDatasetConfig(
        name="toto",
        gcs_path="gs://gresearch/robotics/toto/0.1.0",
        image_key="image",
        image_shape=(480, 640, 3),
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key="state",
        state_dim=2,
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=60,
    ),
    "viola": OXEDatasetConfig(
        name="viola",
        gcs_path="gs://gresearch/robotics/viola/0.1.0",
        image_key="agentview_rgb",
        image_shape=(224, 224, 3),
        instruction_key="natural_language_instruction",
        instruction_in_step=False,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        action_key="world_vector",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=60,
    ),
    "vima_converted_externally_to_rlds": OXEDatasetConfig(
        name="vima_converted_externally_to_rlds",
        gcs_path="gs://gresearch/robotics/vima_converted_externally_to_rlds/0.0.1",
        image_key="image",
        image_shape=(128, 256, 3),
        instruction_key="multimodal_instruction",
        instruction_in_step=True,
        state_key=None,
        state_dim=0,
        allow_missing_state=True,
        action_key="pose0_position",
        action_is_dict=True,
        action_dim=3,
        avg_episode_length=60,
    ),
    "utaustin_mutex": OXEDatasetConfig(
        name="utaustin_mutex",
        gcs_path="gs://gresearch/robotics/utaustin_mutex/0.1.0",
        image_key="image",
        image_shape=(128, 128, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
    "fmb": OXEDatasetConfig(
        name="fmb",
        gcs_path="gs://gresearch/robotics/fmb/0.0.1",
        image_key="image_side_1",
        image_shape=(256, 256, 3),
        instruction_key="language_instruction",
        instruction_in_step=True,
        state_key="state_gripper_pose",
        state_dim=2,
        action_dim=7,
        avg_episode_length=60,
    ),
}


def _normalize_dataset_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _gcs_dataset_segment(gcs_path: str) -> str:
    parts = [p for p in str(gcs_path).split("/") if p]
    if len(parts) < 2:
        return ""
    return parts[-2]


def _build_oxe_dataset_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for key, cfg in OXE_DATASETS.items():
        for candidate in (key, cfg.name, _gcs_dataset_segment(cfg.gcs_path)):
            if not candidate:
                continue
            aliases.setdefault(candidate, key)
            aliases.setdefault(_normalize_dataset_name(candidate), key)
    return aliases


OXE_DATASET_ALIASES = _build_oxe_dataset_aliases()


def resolve_oxe_dataset_key(dataset_name: str) -> Optional[str]:
    if dataset_name in OXE_DATASETS:
        return dataset_name
    aliased = OXE_DATASET_ALIASES.get(dataset_name)
    if aliased is not None:
        return aliased
    return OXE_DATASET_ALIASES.get(_normalize_dataset_name(dataset_name))


def resolve_oxe_dataset_config(dataset_name: str) -> Optional[OXEDatasetConfig]:
    key = resolve_oxe_dataset_key(dataset_name)
    if key is None:
        return None
    return OXE_DATASETS.get(key)


def resolve_nested_key(container, keypath: str):
    """Navigate a nested dict/tensor structure via slash-separated key path.

    Some OXE TFDS datasets use literal keys containing '/' (e.g.
    'future/xyz_residual'). Try a direct lookup first; only treat '/' as a
    nesting delimiter if the full key is not present.
    """
    if "/" not in keypath:
        return container[keypath]
    if keypath in container:
        return container[keypath]
    cur = container
    for part in keypath.split("/"):
        cur = cur[part]
    return cur
