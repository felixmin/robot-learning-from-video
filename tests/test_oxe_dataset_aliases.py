from common.adapters.oxe_shared import (
    resolve_oxe_dataset_config,
    resolve_oxe_dataset_key,
)


def test_resolve_robonet_alias_from_folder_name() -> None:
    assert resolve_oxe_dataset_key("robo_net") == "robonet"
    cfg = resolve_oxe_dataset_config("robo_net")
    assert cfg is not None
    assert cfg.name == "robonet"


def test_resolve_rt1_alias_from_folder_name() -> None:
    assert resolve_oxe_dataset_key("fractal20220817_data") == "rt1"
    cfg = resolve_oxe_dataset_config("fractal20220817_data")
    assert cfg is not None
    assert cfg.name == "rt1"


def test_resolve_alias_normalization() -> None:
    assert resolve_oxe_dataset_key("ROBO-NET") == "robonet"


def test_unknown_dataset_returns_none() -> None:
    assert resolve_oxe_dataset_key("totally_unknown_dataset") is None
    assert resolve_oxe_dataset_config("totally_unknown_dataset") is None

