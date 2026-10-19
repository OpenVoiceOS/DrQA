from importlib.metadata import entry_points

import pytest


def test_extractive_qa_entry_point_registered():
    eps = entry_points(group="opm.agents.extractive_qa")
    names = {ep.name: ep.value for ep in eps}
    if "ovos-drqa-plugin" not in names:
        pytest.skip("package not installed; entry points unavailable in-tree")
    assert names["ovos-drqa-plugin"] == "drqa.opm:DrQAExtractiveQAEngine"


def test_engine_class_importable():
    from ovos_plugin_manager.templates.agents import ExtractiveQAEngine

    from drqa.opm import DrQAExtractiveQAEngine
    assert issubclass(DrQAExtractiveQAEngine, ExtractiveQAEngine)
