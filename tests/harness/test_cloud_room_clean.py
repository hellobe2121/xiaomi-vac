"""Harness tests: the order in which a cloud room clean tries its actions."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.xiaomi_vac.const import (
    CONF_DEVICE_ID,
    CONF_PASS_TOKEN,
    CONF_SERVER,
    CONF_SERVICE_TOKEN,
    CONF_SSECURITY,
    CONF_USER_ID,
    CONF_USERNAME,
)
from custom_components.xiaomi_vac.device import IjaiVacuumDevice
from custom_components.xiaomi_vac.vacuum import _cloud_clean_segments

_SESSION = {
    CONF_USERNAME: "user@example.com",
    CONF_USER_ID: "uid",
    CONF_SSECURITY: "ssec",
    CONF_SERVICE_TOKEN: "svc",
    CONF_PASS_TOKEN: "pass",
    CONF_SERVER: "sg",
    CONF_DEVICE_ID: "did123",
}
_OK = {"code": 0}
_REJECTED = {"code": -1}

_R2211O_START_CUSTOM = (
    "rpc",
    4,
    1,
    [
        {"piid": 1, "value": 18},
        {"piid": 10, "value": '{"selects":[[5,1,1,1,1],[7,1,1,1,2]]}'},
    ],
)
_R2211O_START = ("spec", 2, 3, ["5,7"])
_V17_SET_ROOM_CLEAN = ("spec", 7, 3, ["5,7", 0, 1])
_V17_START = ("spec", 2, 7, ["5,7"])


@pytest.mark.parametrize(
    ("model", "responses", "expected_calls", "all_rejected"),
    [
        ("dreame.vacuum.r2211o", [_OK], [_R2211O_START_CUSTOM], False),
        (
            "dreame.vacuum.r2211o",
            [_REJECTED, _OK],
            [_R2211O_START_CUSTOM, _R2211O_START],
            False,
        ),
        (
            "dreame.vacuum.r2211o",
            [_REJECTED, _REJECTED],
            [_R2211O_START_CUSTOM, _R2211O_START],
            True,
        ),
        ("ijai.vacuum.v17", [_OK], [_V17_SET_ROOM_CLEAN], False),
        (
            "ijai.vacuum.v17",
            [_REJECTED, _OK],
            [_V17_SET_ROOM_CLEAN, _V17_START],
            False,
        ),
    ],
)
def test_cloud_room_clean_tries_actions_in_order_and_stops_at_first_success(
    model: str, responses: list, expected_calls: list, all_rejected: bool
) -> None:
    with patch("custom_components.xiaomi_vac.device.MiotDevice"):
        device = IjaiVacuumDevice("host", "token", model)
    calls: list[tuple] = []
    pending = iter(responses)

    def _rpc(server, did, siid, aiid, in_params):
        calls.append(("rpc", siid, aiid, in_params))
        return next(pending)

    def _spec(server, did, siid, aiid, params):
        calls.append(("spec", siid, aiid, params))
        return next(pending)

    cloud = MagicMock()
    cloud.cloud_rpc_action.side_effect = _rpc
    cloud.cloud_action.side_effect = _spec

    with patch("custom_components.xiaomi_vac.vacuum.XiaomiCloud", return_value=cloud):
        if all_rejected:
            with pytest.raises(ValueError):
                _cloud_clean_segments(dict(_SESSION), device, [5, 7])
        else:
            _cloud_clean_segments(dict(_SESSION), device, [5, 7])

    assert calls == expected_calls
