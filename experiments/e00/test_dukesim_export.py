import numpy as np
import pytest

import dukesim_export as d

IDS = [0, 1, 4, 6, 26, 60, 120]


@pytest.fixture
def pkg(tmp_path):
    tab = d.catalogue_table()
    lab = np.random.default_rng(0).choice(IDS, (4, 6, 8)).astype(np.uint8)
    d.write_package(tmp_path, lab, 0.25, tab, sid="t1")
    return tmp_path, tab


def status(p):
    return {k: v["status"] for k, v in d.validate_package(p).items()}


def test_roundtrip_pass(pkg):
    p, _ = pkg
    st = status(p)
    assert [k for k, v in st.items() if v == "FAIL"] == [], d.validate_package(p)
    assert st["V3"] == st["V11"] == st["V12"] == "PASS" and st["V14"] == "SKIP"


def test_label_not_in_table(pkg):
    p, _ = pkg
    f = next(p.glob("phantom_*.bin"))
    b = bytearray(f.read_bytes()); b[0] = 99; f.write_bytes(bytes(b))
    assert status(p)["V3"] == "FAIL"


def test_wrong_size(pkg):
    p, _ = pkg
    f = next(p.glob("phantom_*.bin"))
    f.write_bytes(f.read_bytes()[:-1])
    assert status(p)["V1"] == "FAIL"


def test_density_nonpositive(tmp_path):
    tab = d.catalogue_table(); tab[4]["density"] = -1.0
    d.write_package(tmp_path, np.zeros((2, 2, 2), np.uint8), 0.25, tab)
    st = status(tmp_path)
    assert st["V7"] == "FAIL"


def test_lac_wrong_shape_and_values(pkg):
    p, _ = pkg
    f = p / "lin_att_120kV_121materials.bin"
    f.write_bytes(f.read_bytes()[:-4])
    assert status(p)["V12"] == "FAIL"
    d.write_tables(p, d.catalogue_table(), "t1")
    a = np.fromfile(f, "<f4"); a[4 * 120 + 5] *= 1.01; a.tofile(f)  # row 4 (muscle) perturbed
    assert status(p)["V12"] == "FAIL"


def test_lac_wrong_dtype(pkg):
    p, _ = pkg
    f = p / "lin_att_120kV_121materials.bin"
    np.fromfile(f, "<f4").astype("<f8").tofile(f)  # double: 2x size
    assert status(p)["V12"] == "FAIL"


def test_unknown_composition_rejected(tmp_path):
    tab = d.catalogue_table(); tab[4]["fractions"]["6"] += 0.1
    with pytest.raises(ValueError):
        d.write_package(tmp_path, np.zeros((2, 2, 2), np.uint8), 0.25, tab)
