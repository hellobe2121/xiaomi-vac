"""Pure-tier tests for the map fetch cycle diagnostic record.

The record is the tested artefact: what MapFetcher.fetch() leaves behind per
slot, how the active map id is resolved and by which trust step, and the shape
of the whole-cycle record. No homeassistant import; no network.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from xvac.cloud.connector import CloudUnreachable
from xvac.map import MapFetcher, SessionExpired, _vector_source
from xvac.map_diagnostics import (
    MapCycleRecord,
    SlotAttempt,
    describe_map_capability,
    describe_map_key_owner,
    pick_map_key_owner,
    resolve_active_id,
)


class FakeCloud:
    """Duck-typed XiaomiCloud: canned URL and blob."""

    def __init__(self, *, url="http://x", blob=b""):
        self.url = url
        self.blob = blob
        self.calls = []

    def cloud_get_prop(self, server, did, siid, piid):
        return {"result": [{"value": None}]}

    def get_user_device_data(self, server, did, siid, piid, limit=2, time_start=0):
        return None

    def map_url(self, server, did, slot, endpoint):
        self.calls.append(("map_url", slot))
        return self.url

    def download(self, url):
        self.calls.append(("download", url))
        return self.blob


def _fetcher(cloud):
    return MapFetcher(
        cloud, server="de", user_id="1", device_id="2",
        model="dreame.vacuum.p2008", mac="AA", wifi_sn="SN", parser_brand="dreame")


def _md(*, image):
    return SimpleNamespace(
        image=image, calibration=lambda: [], rooms={}, charger=None,
        vacuum_position=None, vacuum_room=None, vacuum_room_name=None,
        zones=[], no_go_areas=[], no_mopping_areas=[], walls=[],
        path=None, goto=None,
    )


def _with_parser(fetcher, *, unpack, parse):
    fetcher._parser = SimpleNamespace(unpack_map=unpack, parse=parse)


def _renderable_md():
    return _md(image=SimpleNamespace(is_empty=False, data=Image.new("RGB", (8, 8), "white")))


# --- per-slot record from MapFetcher.fetch() -------------------------------
def test_fetch_records_no_url_and_still_raises_session_expired():
    f = _fetcher(FakeCloud(url=None))
    with pytest.raises(SessionExpired):
        f.fetch("0")
    attempt = f.last_attempt
    assert attempt.slot == "0"
    assert attempt.url_obtained is False
    assert attempt.blob_bytes is None
    assert attempt.outcome == "no_url"


def test_fetch_records_unreachable_and_reraises():
    cloud = FakeCloud()

    def _timeout(*a):
        raise CloudUnreachable("tw timed out")
    cloud.map_url = _timeout
    f = _fetcher(cloud)
    with pytest.raises(CloudUnreachable):
        f.fetch("0")
    assert f.last_attempt.url_obtained is False
    assert f.last_attempt.outcome == "unreachable"


def test_fetch_records_empty_download_as_none_return():
    f = _fetcher(FakeCloud(blob=b""))
    assert f.fetch("1") is None
    attempt = f.last_attempt
    assert attempt.slot == "1"
    assert attempt.url_obtained is True
    assert attempt.blob_bytes == 0
    assert attempt.outcome == "empty_download"


def test_fetch_records_undecryptable_blob():
    f = _fetcher(FakeCloud(blob=b"garbage-bytes"))

    def _boom(raw, **kw):
        raise ValueError("Padding is incorrect")

    _with_parser(f, unpack=_boom, parse=lambda u: None)
    assert f.fetch() is None
    attempt = f.last_attempt
    assert attempt.url_obtained is True
    assert attempt.blob_bytes == 13
    assert attempt.outcome == "undecryptable"


def test_fetch_logs_only_first_16_bytes_of_undecryptable_blob(caplog):
    f = _fetcher(FakeCloud(blob=b'{"version":2,"d' + b"X" + b"TAILMARKER"))

    def _boom(raw, **kw):
        raise ValueError("Expecting value")

    _with_parser(f, unpack=_boom, parse=lambda u: None)
    with caplog.at_level("DEBUG"):
        assert f.fetch() is None
    assert "Could not decrypt map at slot 0" in caplog.text
    assert "b'{\"version\":2,\"dX'" in caplog.text
    assert "TAILMARKER" not in caplog.text


def test_fetch_records_blob_that_does_not_parse():
    f = _fetcher(FakeCloud(blob=b"x" * 40))

    def _bad_parse(unpacked):
        raise ValueError("bad frame")

    _with_parser(f, unpack=lambda raw, **kw: b"u", parse=_bad_parse)
    assert f.fetch() is None
    attempt = f.last_attempt
    assert attempt.blob_bytes == 40
    assert attempt.outcome == "parse_rejected"


def test_fetch_records_parse_with_no_image():
    f = _fetcher(FakeCloud(blob=b"x" * 40))
    _with_parser(f, unpack=lambda raw, **kw: b"u", parse=lambda u: _md(image=None))
    assert f.fetch() is None
    assert f.last_attempt.outcome == "empty_render"


def test_fetch_records_blob_that_parses_to_a_render():
    f = _fetcher(FakeCloud(blob=b"x" * 40))
    _with_parser(f, unpack=lambda raw, **kw: b"u", parse=lambda u: _renderable_md())
    result = f.fetch()
    assert result is not None
    attempt = f.last_attempt
    assert attempt.url_obtained is True
    assert attempt.blob_bytes == 40
    assert attempt.outcome == "rendered"


def test_fetch_record_never_contains_the_url_value():
    f = _fetcher(FakeCloud(url="https://signed.example/secret-token-abc", blob=b""))
    f.fetch()
    assert "secret-token-abc" not in json.dumps(f.last_attempt.as_dict())


# --- active-map resolution: which ADR-0013 trust step decided ---------------
_SINGLE = 0


def _resolve(*, decoded=(), active_meta=None, mqtt=None, has_map_list=True, maps_meta=()):
    return resolve_active_id(
        [SimpleNamespace(map_id=i) for i in decoded], active_meta, mqtt,
        has_map_list, list(maps_meta), single_map_id=_SINGLE)


def test_resolution_prefers_the_live_blobs_embedded_id():
    assert _resolve(decoded=[555], active_meta={"id": 111, "cur": True}) == (555, "blob")


def test_resolution_uses_map_list_cur_when_blob_carries_no_id():
    assert _resolve(decoded=[None], active_meta={"id": "111", "cur": True}) == (111, "map_list")


def test_resolution_uses_single_map_key_only_when_profile_has_no_map_list():
    assert _resolve(has_map_list=False, maps_meta=[]) == (_SINGLE, "single_map")


def test_empty_map_list_read_on_multi_map_device_is_unresolved():
    assert _resolve(has_map_list=True, maps_meta=[]) == (None, None)


def test_resolution_falls_to_mqtt_id_when_map_list_read_failed():
    assert _resolve(mqtt=777) == (777, "mqtt")


def test_unparseable_map_list_id_is_skipped():
    assert _resolve(active_meta={"id": "abc", "cur": True}, mqtt=9) == (9, "mqtt")


# --- whole-cycle record ------------------------------------------------------
def _record(**kw):
    return MapCycleRecord(parser_key="ijai", map_capability=None, **kw)


def _slots(*outcomes):
    """(url_obtained, blob_bytes, outcome) per slot."""
    return [
        SlotAttempt(slot=str(i), url_obtained=u, blob_bytes=b, outcome=o)
        for i, (u, b, o) in enumerate(outcomes)
    ]


def test_cycle_where_cloud_refused_every_url_reads_as_session_expired():
    rec = _record(slots=_slots((False, None, "no_url"), (False, None, "no_url")))
    out = rec.as_dict()
    assert out["session_expired"] is True
    assert out["url_obtained"] is False
    assert out["rendered"] is False


def test_cycle_where_cloud_was_unreachable_is_not_session_expired():
    rec = _record(slots=_slots((False, None, "unreachable")))
    assert rec.as_dict()["session_expired"] is False


def test_cycle_with_url_but_empty_blob_is_not_session_expired():
    rec = _record(slots=_slots((True, 0, "empty_download"), (True, 0, "empty_download")))
    out = rec.as_dict()
    assert out["session_expired"] is False
    assert out["url_obtained"] is True
    assert out["rendered"] is False


def test_cycle_with_blob_that_does_not_parse_is_not_rendered():
    rec = _record(slots=_slots((True, 52000, "parse_rejected"), (True, 0, "empty_download")))
    out = rec.as_dict()
    assert out["rendered"] is False
    assert out["slots"][0]["blob_bytes"] == 52000


def test_cycle_with_one_readable_slot_is_rendered_even_if_the_other_is_dead():
    rec = _record(slots=_slots((False, None, "no_url"), (True, 52000, "rendered")))
    out = rec.as_dict()
    assert out["rendered"] is True
    assert out["session_expired"] is False


def test_cycle_before_any_fetch_is_not_reported_as_session_expired():
    assert _record().as_dict()["session_expired"] is False


def test_cycle_served_is_cache_when_nothing_decoded_but_cache_held_the_map():
    rec = _record()
    rec.set_served(decoded=False, have_result=True)
    assert rec.as_dict()["served"] == "cache"


def test_cycle_served_is_live_when_a_slot_decoded():
    rec = _record()
    rec.set_served(decoded=True, have_result=True)
    assert rec.as_dict()["served"] == "live"


def test_cycle_served_is_none_on_cold_start():
    rec = _record()
    rec.set_served(decoded=False, have_result=False)
    assert rec.as_dict()["served"] == "none"


def test_cycle_record_is_json_serialisable_and_carries_resolution():
    rec = MapCycleRecord(
        parser_key="xiaomi", map_capability={"service": 10, "declared": {}},
        slots=_slots((True, 10, "rendered")), resolved_map_id=42, resolved_by="blob")
    out = json.loads(json.dumps(rec.as_dict()))
    assert out["parser_key"] == "xiaomi"
    assert out["resolved_map_id"] == 42
    assert out["resolved_by"] == "blob"


# --- profile map capability shape --------------------------------------------
def test_capability_shape_lists_only_declared_ids():
    from spec.types import Action, MapCapability, Prop

    cap = MapCapability(service=10, map_num=Prop(10, 4), get_map_list=Action(10, 3))
    assert describe_map_capability(cap) == {
        "service": 10, "declared": {"map_num": "10.4", "get_map_list": "10.3"}}


def test_capability_shape_is_none_for_profile_without_map():
    assert describe_map_capability(None) is None


def test_capability_shape_of_every_registered_profile_is_serialisable():
    from spec.registry import MODEL_PROFILES

    for profile in MODEL_PROFILES.values():
        json.dumps(describe_map_capability(profile.map))


# --- map key owner source (#51) --------------------------------------------
def test_map_key_owner_prefers_device_reported_uid():
    assert pick_map_key_owner("222", "111", "333") == "333"
    assert pick_map_key_owner("222", "111", None) == "222"
    assert pick_map_key_owner(None, "111", None) == "111"
    assert pick_map_key_owner(None, 111, None) == "111"


def test_describe_map_key_owner_names_device_source():
    assert describe_map_key_owner("ijai", "222", "111", "333") == ("device_7_45", False)
    assert describe_map_key_owner("ijai", "222", "111", "111") == ("device_7_45", True)
    assert describe_map_key_owner("ijai", "222", "111", None) == ("owner_uid", False)
    assert describe_map_key_owner("ijai", None, "111", None) == ("user_id", None)
    assert describe_map_key_owner("dreame", "222", "111", "333") == (None, None)


# --- vector source after parse (#51 room fills) ----------------------------
def test_vector_source_uses_parser_robot_map_for_ijai():
    parser = SimpleNamespace(robot_map=SimpleNamespace(SerializeToString=lambda: b"labelled"))
    assert _vector_source(parser, b"raw", True) == b"labelled"


def test_vector_source_keeps_unpacked_when_not_ijai_or_no_robot_map():
    parser = SimpleNamespace(robot_map=SimpleNamespace(SerializeToString=lambda: b"labelled"))
    assert _vector_source(parser, b"raw", False) == b"raw"
    assert _vector_source(SimpleNamespace(), b"raw", True) == b"raw"


# --- ijai key type fallback (#51 hex key) ----------------------------------
def _ijai_fetcher(cloud):
    return MapFetcher(
        cloud, server="de", user_id="1", device_id="2",
        model="xiaomi.vacuum.d106gl", mac="AA:BB:CC:DD:EE:FF", wifi_sn="12345/ABCDE12345FGHIJ",
        parser_brand="ijai")


def _encrypt_ijai(plain: bytes, *, hex_key: bool, **kw) -> bytes:
    """Encrypt a RobotMap payload the way the cloud does, for one key type."""
    import zlib

    from Crypto.Cipher import AES
    from Crypto.Hash import MD5
    from Crypto.Util.Padding import pad
    from vacuum_map_parser_ijai import aes_decryptor

    mac_part = "".join(kw["device_mac"].lower().split(":"))
    suffix = kw["model"].split(".")[-1][-4:]
    aeskey = aes_decryptor.aes_encrypt(
        "+".join([kw["wifi_sn"], kw["owner_id"], kw["device_id"]]), mac_part + suffix)
    digest = MD5.new(aeskey.encode("utf-8")).hexdigest()
    key = bytes.fromhex(digest) if hex_key else digest[8:-8].upper().encode("utf-8")
    body = zlib.compress(plain).hex().encode("utf-8")
    return AES.new(key, AES.MODE_ECB).encrypt(pad(body, AES.block_size, "pkcs7"))


@pytest.mark.parametrize("hex_key", [True, False])
def test_ijai_unpack_tries_both_key_types(hex_key):
    f = _ijai_fetcher(FakeCloud())
    blob = _encrypt_ijai(b"\x08\x01robotmap", hex_key=hex_key, **f._unpack_kw)

    assert f._unpack(blob) == b"\x08\x01robotmap"
    assert f.last_key_type == ("hex" if hex_key else "text")


def test_ijai_unpack_raises_when_neither_key_fits():
    f = _ijai_fetcher(FakeCloud())
    with pytest.raises(Exception):
        f._unpack(b"\x00" * 32)
