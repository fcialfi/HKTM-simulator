import datetime
import json

import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

import licensing  # noqa: E402

TODAY = datetime.date(2026, 10, 9)


@pytest.fixture
def keys(tmp_path):
    key = Ed25519PrivateKey.generate()
    pub = tmp_path / "pub.pem"
    pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo))
    return key, pub


def _write(tmp_path, lic):
    path = tmp_path / "license.key"
    path.write_text(json.dumps(lic), encoding="utf-8")
    return path


def _check(tmp_path, pub, lic_path, today=TODAY):
    return licensing.check_license(today=today, license_path=str(lic_path), public_key_path=str(pub),
                                   state_path=str(tmp_path / "state.json"), enforce_from_source=True)


def test_valid_license(tmp_path, keys):
    key, pub = keys
    path = _write(tmp_path, licensing.sign_license(key, "ACME", datetime.date(2027, 1, 31)))
    st = _check(tmp_path, pub, path)
    assert st.valid and st.enforced
    assert st.customer == "ACME" and st.days_left == (datetime.date(2027, 1, 31) - TODAY).days
    assert not st.expiring_soon


def test_last_day_is_valid_next_day_expired(tmp_path, keys):
    key, pub = keys
    path = _write(tmp_path, licensing.sign_license(key, "ACME", TODAY))
    assert _check(tmp_path, pub, path).valid
    st = _check(tmp_path, pub, path, today=TODAY + datetime.timedelta(days=1))
    assert not st.valid and "expired" in st.message


def test_expiring_soon_flag(tmp_path, keys):
    key, pub = keys
    path = _write(tmp_path, licensing.sign_license(key, "ACME", TODAY + datetime.timedelta(days=10)))
    assert _check(tmp_path, pub, path).expiring_soon


def test_edited_expiry_is_rejected(tmp_path, keys):
    key, pub = keys
    lic = licensing.sign_license(key, "ACME", datetime.date(2026, 1, 1))
    lic["expires"] = "2099-12-31"
    st = _check(tmp_path, pub, _write(tmp_path, lic))
    assert not st.valid and "signature" in st.message


def test_license_from_another_key_is_rejected(tmp_path, keys):
    _, pub = keys
    other = Ed25519PrivateKey.generate()
    st = _check(tmp_path, pub, _write(tmp_path, licensing.sign_license(other, "ACME", datetime.date(2099, 1, 1))))
    assert not st.valid


def test_missing_and_damaged_files(tmp_path, keys):
    _, pub = keys
    st = _check(tmp_path, pub, tmp_path / "license.key")
    assert not st.valid and "No licence found" in st.message
    (tmp_path / "license.key").write_text("not json", encoding="utf-8")
    assert not _check(tmp_path, pub, tmp_path / "license.key").valid
    (tmp_path / "license.key").write_text('{"customer": "x"}', encoding="utf-8")
    assert "damaged" in _check(tmp_path, pub, tmp_path / "license.key").message


def test_clock_set_back_is_rejected(tmp_path, keys):
    key, pub = keys
    path = _write(tmp_path, licensing.sign_license(key, "ACME", datetime.date(2027, 1, 1)))
    assert _check(tmp_path, pub, path, today=TODAY).valid
    assert _check(tmp_path, pub, path, today=TODAY - datetime.timedelta(days=1)).valid  # tolerance
    st = _check(tmp_path, pub, path, today=TODAY - datetime.timedelta(days=30))
    assert not st.valid and "set back" in st.message


def test_not_enforced_without_public_key_or_from_source(tmp_path, keys):
    key, pub = keys
    st = licensing.check_license(today=TODAY, license_path=str(tmp_path / "none"),
                                 public_key_path=str(tmp_path / "missing.pem"),
                                 state_path=str(tmp_path / "s.json"), enforce_from_source=True)
    assert st.valid and not st.enforced
    st = licensing.check_license(today=TODAY, license_path=str(tmp_path / "none"),
                                 public_key_path=str(pub), state_path=str(tmp_path / "s.json"))
    assert st.valid and not st.enforced  # running from source
