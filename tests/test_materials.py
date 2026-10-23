import copy
import gc
import json

import pytest

from fdtd import Material
from fdtd.exceptions import MaterialExists, MaterialNotFound


def test_failed_duplicate_does_not_remove_registered_material():
    original = Material("duplicate-keeps-original", eps_r=3)
    with pytest.raises(MaterialExists):
        Material(original.name, eps_r=4)
    gc.collect()
    assert Material[original.name] is original
    clone = copy.deepcopy(original)
    clone._delete()
    assert Material[original.name] is original


def test_loading_is_atomic_and_repeatable(tmp_path):
    path = tmp_path / "materials.json"
    path.write_text(
        json.dumps([{"name": "atomic-first"}, {"name": "atomic-bad", "eps_r": 0}])
    )
    with pytest.raises(ValueError):
        Material.load(path)
    with pytest.raises(MaterialNotFound):
        Material["atomic-first"]
    path.write_text(json.dumps([{"name": "atomic-first", "eps_r": 2}]))
    first = Material.load(path)
    assert Material.load(path) == first
    assert first[0].sigma_e == 0
    path.write_text(json.dumps([{"name": "atomic-first", "eps_r": 3}]))
    with pytest.raises(MaterialExists):
        Material.load(path)


def test_material_names_cannot_change_after_registration():
    material = Material("stable-name")
    with pytest.raises(AttributeError):
        material.name = "changed"
