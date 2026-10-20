"""Extract a vector/grid map representation from the decoded ijai blob.

See docs/dev/map-pipeline.md for the per-brand map pipeline details.
"""
from __future__ import annotations

import base64
import json
import logging
import zlib
from collections.abc import Callable
from typing import Any

import vacuum_map_parser_ijai.RobotMap_pb2 as RobotMap

_LOGGER = logging.getLogger(__name__)


def _rle(grid: bytes) -> list[int]:
    """Run-length encode the grid as a flat [value, count, value, count, ...]."""
    out: list[int] = []
    if not grid:
        return out
    prev = grid[0]
    run = 1
    for b in grid[1:]:
        if b == prev and run < 0xFFFFFFFF:
            run += 1
        else:
            out.append(prev)
            out.append(run)
            prev = b
            run = 1
    out.append(prev)
    out.append(run)
    return out


def _pt(p: Any, to_m: Callable[[Any], Any] = lambda v: v) -> dict[str, float]:
    d = {"x": to_m(p.x), "y": to_m(p.y)}
    a = getattr(p, "phi", None)
    if a is not None:
        d["a"] = a  # heading in degrees -- never a length, never scaled
    return d


# --- room-contour tracing -------------------------------------------------
# The firmware's roomChain is a coarse ~8-vertex cartoon of each room. The
# labelled occupancy grid is ground truth: one byte/cell, room id 10-59 (or the
# same id + 50 when that room is "selected"). We trace the EXACT cell outline of
# each room along grid lines — no smoothing — so the card renders pixel-true
# room areas (the staircase is sub-pixel at card size, exactly like the raw blob
# the official app draws). Each room becomes one {id, rings:[[[col,row],...]]}
# entry; multiple rings (disconnected pieces + furniture holes) render as a
# single even-odd path, so holes punch through. id == grid label == md.rooms key.

ROOM_MIN, ROOM_MAX = 10, 59
SELECTED_OFFSET = 50  # selected-room cell value = base id + 50 (60-109)


def _label_of(v: int) -> int | None:
    """Map a cell value to its base room id, or None if it isn't a room."""
    if ROOM_MIN <= v <= ROOM_MAX:
        return v
    if ROOM_MAX + SELECTED_OFFSET >= v >= ROOM_MIN + SELECTED_OFFSET:
        return v - SELECTED_OFFSET
    return None


def _trace_mask(mask: set) -> list[list[list[int]]]:
    """Trace exact boundary loops of a set of (col,row) cells along grid lines.

    Each empty-neighbour side of a filled cell is a directed unit edge wound so
    the interior stays on the right. Edges stitch end-to-start into closed loops.

    At a "pinch" vertex — where the boundary touches itself at a single corner
    (two cells meeting only diagonally, which is everywhere in noisy scan data) —
    a corner is the start of TWO edges. A plain start->end map loses one, the
    walk dead-ends, and the loop closes with a stray diagonal chord. So we keep a
    multimap and, at every vertex, leave by the most-clockwise turn relative to
    how we arrived. That consistently separates the touching loops instead of
    tangling them.
    """
    adj: dict[tuple, list[tuple]] = {}
    remaining: set[tuple] = set()  # (start, end) edges not yet walked
    for (c, r) in mask:
        for a, b in (
            ((c, r), (c + 1, r)) if (c, r - 1) not in mask else (None, None),
            ((c + 1, r), (c + 1, r + 1)) if (c + 1, r) not in mask else (None, None),
            ((c + 1, r + 1), (c, r + 1)) if (c, r + 1) not in mask else (None, None),
            ((c, r + 1), (c, r)) if (c - 1, r) not in mask else (None, None),
        ):
            if a is not None:
                adj.setdefault(a, []).append(b)
                remaining.add((a, b))

    def _next(cur: tuple, din: tuple, cands: list[tuple]) -> tuple:
        # pick the outgoing edge that turns most clockwise from incoming dir din.
        # rank: right turn (0) < straight (1) < left (2) < u-turn (3); interior is
        # on the right, so hugging clockwise keeps each loop separate at a pinch.
        def rank(en: tuple) -> int:
            dx, dy = en[0] - cur[0], en[1] - cur[1]
            cross = din[0] * dy - din[1] * dx   # >0 = left turn (y-down grid)
            dot = din[0] * dx + din[1] * dy
            if cross < 0:
                return 0                        # right turn
            if cross == 0:
                return 1 if dot > 0 else 3      # straight vs u-turn
            return 2                            # left turn
        return min(cands, key=rank)

    loops = []
    while remaining:
        start, end = next(iter(remaining))
        remaining.discard((start, end))
        loop = [list(start), list(end)]
        prev, cur = start, end
        while cur != start:
            din = (cur[0] - prev[0], cur[1] - prev[1])
            cands = [e for e in adj.get(cur, ()) if (cur, e) in remaining]
            if not cands:
                break
            nxt = _next(cur, din, cands)
            remaining.discard((cur, nxt))
            prev, cur = cur, nxt
            loop.append(list(cur))
        if cur == start:
            loop.pop()                          # drop the repeated start vertex
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _drop_collinear(loop: list[list[int]]) -> list[list[int]]:
    """Drop vertices mid-run along a straight axis edge — keeps the staircase
    shape pixel-identical but shrinks long flat walls to their two endpoints."""
    n = len(loop)
    if n < 3:
        return loop
    out = []
    for i in range(n):
        a, b, cc = loop[(i - 1) % n], loop[i], loop[(i + 1) % n]
        # cross product (b-a)x(c-b); 0 => b lies on the a->c line, so drop it
        if (b[0] - a[0]) * (cc[1] - b[1]) - (b[1] - a[1]) * (cc[0] - b[0]) != 0:
            out.append(b)
    return out or loop


def _signed_area(loop: list[list[int]]) -> float:
    """Twice the signed polygon area (shoelace); sign encodes winding."""
    s = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s


def _chains_from_masks(masks: dict[int, set]) -> list[dict[str, Any]]:
    """Trace each label's cell mask into its outer-boundary rings.

    Split out of `trace_room_chains` so a parser whose grid uses a DIFFERENT
    label alphabet can reuse the identical tracing, winding and ring-selection
    rules — `extract_json_grid` keys its masks by real room id, which for the
    xiaomi JSON family sits outside the ijai 10-59 band.
    """
    chains: list[dict[str, Any]] = []
    for lab in sorted(masks):
        loops = _trace_mask(masks[lab])
        if not loops:
            continue
        # Outer boundaries share the winding of the largest loop; holes wind the
        # other way. Keep outer rings only (one per disconnected room piece).
        outer = _signed_area(max(loops, key=lambda L: abs(_signed_area(L)))) >= 0
        rings = [
            _drop_collinear(loop)
            for loop in loops
            if (_signed_area(loop) >= 0) == outer and len(loop) >= 3
        ]
        if rings:
            chains.append({"id": lab, "rings": rings})
    return chains


def trace_room_chains(grid: bytes, w: int, h: int) -> list[dict[str, Any]]:
    """Trace exact room outlines from the labelled grid (row-major, w*h cells).

    Follows what proven renderers (Valetudo) do: each room is a SOLID area,
    obstacles are a separate layer drawn on top — never holes punched into the
    fill. So we keep only the OUTER boundary of each room component and drop
    interior holes (furniture/wall cells). Outline is exact (no smoothing); the
    card fills it with no stroke, so the staircase is sub-pixel like the raw blob.
    """
    masks: dict[int, set] = {}
    for r in range(h):
        base = r * w
        for c in range(w):
            lab = _label_of(grid[base + c])
            if lab is not None:
                masks.setdefault(lab, set()).add((c, r))
    return _chains_from_masks(masks)


def extract_grid(unpacked: bytes) -> dict[str, Any]:
    """Pull the labelled grid + transform + room chains from the unpacked blob.

    Vector overlays already in metres (path/charger/walls/...) come from the
    parser's MapData; this returns only what the parser discards: the raw grid,
    the cell<->metre bounds, and the room boundary chains.
    """
    rm = RobotMap.RobotMap()
    rm.ParseFromString(unpacked)
    h = rm.mapHead
    grid = bytes(rm.mapData.mapData)

    # Prefer contours traced from the labelled grid (true wall-aligned edges).
    # Fall back to the firmware's coarse roomChain only if the grid has no
    # labelled rooms (e.g. a fresh/quick map with vector chains but no fill).
    chains = trace_room_chains(grid, h.sizeX, h.sizeY)
    if not chains:
        for room in rm.roomChain:
            pts = [[p.x, p.y] for p in room.points]
            if pts:
                chains.append({"id": room.roomId, "rings": [pts]})

    return {
        "map_id": h.mapHeadId,  # identifies the physical map (dedupe across slots)
        "size": {"x": h.sizeX, "y": h.sizeY},
        # cell (col,row) centre in metres:
        #   mx = minX + (col + 0.5) * (maxX - minX) / sizeX
        #   my = minY + (row + 0.5) * (maxY - minY) / sizeY
        # and the inverse for metre -> cell. Overlays use the SAME transform so
        # they line up with the grid regardless of render orientation.
        "bounds": {"minX": h.minX, "minY": h.minY, "maxX": h.maxX, "maxY": h.maxY},
        "resolution": h.resolution,
        "grid_rle": _rle(grid),  # row-major, len == sizeX*sizeY when expanded
        "room_chains": chains,   # grid-cell polygons; may be empty
        # legend so the card can theme cell types without magic numbers
        "legend": {
            "outside": 0, "floor": 1, "new_area": 2, "wall": 255,
            "room_min": 10, "room_max": 59,
            "selected_room_min": 60, "selected_room_max": 109,
        },
    }


def _empty_grid() -> dict[str, Any]:
    """Grid-less contract for brands whose unpacked blob is NOT an ijai protobuf.

    The labelled occupancy grid + room contours are ijai-only (`extract_grid`).
    Other brands (xiaomi/dreame/viomi) ship the rendered PNG + vector overlays
    (already in metres) instead; the card overlays those using the attribute
    `calibration_points` rather than this grid. Same key shape, just empty.
    """
    return {
        "map_id": None,
        "size": None,
        "bounds": None,
        "resolution": None,
        "grid_rle": [],
        "room_chains": [],
        "legend": {
            "outside": 0, "floor": 1, "new_area": 2, "wall": 255,
            "room_min": 10, "room_max": 59,
            "selected_room_min": 60, "selected_room_max": 109,
        },
    }


# --- xiaomi JSON-map grid -------------------------------------------------
# The JSON-map family (ov71gl, ov81gl, d109gl, ...) ships the same kind of
# labelled occupancy grid as ijai, only packed differently: `map_data` is
# base64 + zlib over ONE BYTE PER CELL, row-major, row 0 == origin_y — north
# increases with the row index, exactly the convention `extract_grid`
# documents, so traced chains land in the same space the card expects.
# Cell alphabet (vacuum_map_parser_xiaomi `_normalize_json_map_pixels`):
# 0 = unknown/outside, 1-2 = free floor, 3-63 = room grid_id, >63 = wall.
JSON_ROOM_MIN, JSON_ROOM_MAX = 3, 63


def extract_json_grid(payload: Any, *, units_per_metre: float = 1.0) -> dict[str, Any]:
    """Trace room contours from a decrypted xiaomi JSON map payload.

    `payload` is the JSON string `MapFetcher._unpack` returns for this brand
    (or an already-parsed dict). Returns the same key shape as `extract_grid`.

    Chain vertices are grid-line (col, row) pairs, which the card turns into
    metres as `minX + col * resolution` — so `bounds` and `resolution` are
    converted to metres here with the SAME `units_per_metre` divisor the
    overlays use (this parser reports both in millimetres).

    Rooms are keyed by the user-facing room id, not the raw cell value, so a
    chain id matches `md.rooms` / the `clean_segment` segment id.

    `grid_rle` + `size` are emitted ONLY when every room label also falls in
    the card's room band (ROOM_MIN..ROOM_MAX). The card reads raster cells
    with that band hard-coded and then looks the label up in the `rooms` list
    by id; a grid whose labels are real room ids OUTSIDE the band would paint
    a fully transparent raster OVER the traced fills and hide every room. On
    ov71gl the room ids are 3-7, so it takes the chains-only path and the card
    draws the traced polygons directly. `map_id` deliberately stays None: the
    coordinator trusts a blob-embedded id as ground truth when resolving which
    physical map a cycle belongs to, and that guarantee is ijai-only.
    """
    out = _empty_grid()
    try:
        data = (json.loads(payload)
                if isinstance(payload, (str, bytes, bytearray)) else payload)
    except ValueError as ex:
        _LOGGER.debug("xiaomi JSON grid: payload is not valid JSON (%s)", ex)
        return out
    if not isinstance(data, dict):
        _LOGGER.debug("xiaomi JSON grid: payload is %s, not an object", type(data).__name__)
        return out

    width, height, encoded = data.get("width"), data.get("height"), data.get("map_data")
    if not width or not height or not encoded:
        _LOGGER.debug("xiaomi JSON grid: payload carries no map_data/width/height")
        return out
    try:
        w, h = int(width), int(height)
        cells = zlib.decompress(base64.b64decode(encoded))
    except Exception as ex:  # noqa: BLE001
        _LOGGER.debug("xiaomi JSON grid: map_data is not base64+zlib (%s)", ex)
        return out
    if len(cells) < w * h:
        _LOGGER.debug("xiaomi JSON grid: %d cells for a %dx%d map", len(cells), w, h)
        return out

    # With no mapping the grid_id IS the room id, the same fallback
    # vacuum_map_parser_xiaomi applies.
    grid_to_room = json_room_map(data)

    masks: dict[int, set] = {}
    for r in range(h):
        base = r * w
        for c in range(w):
            v = cells[base + c]
            if JSON_ROOM_MIN <= v <= JSON_ROOM_MAX:
                masks.setdefault(grid_to_room.get(v, v), set()).add((c, r))
    if not masks:
        _LOGGER.debug("xiaomi JSON grid: no labelled room cells in a %dx%d map", w, h)
        return out

    try:
        res = float(data.get("resolution", 50)) / units_per_metre
        min_x = float(data.get("origin_x", 0)) / units_per_metre
        min_y = float(data.get("origin_y", 0)) / units_per_metre
    except (TypeError, ValueError, ZeroDivisionError) as ex:
        _LOGGER.debug("xiaomi JSON grid: invalid geometry metadata (%s)", ex)
        return _empty_grid()
    out["bounds"] = {"minX": min_x, "minY": min_y,
                     "maxX": min_x + w * res, "maxY": min_y + h * res}
    out["resolution"] = res
    out["room_chains"] = _chains_from_masks(masks)

    if all(ROOM_MIN <= lab <= ROOM_MAX for lab in masks):
        # Room ids double as valid card room-band values here, so the raster
        # grid is safe to ship too: relabel the JSON alphabet to the legend.
        norm = bytearray(w * h)
        for i in range(w * h):
            v = cells[i]
            if v in (1, 2):
                norm[i] = v                                 # floor / new area
            elif JSON_ROOM_MIN <= v <= JSON_ROOM_MAX:
                norm[i] = grid_to_room.get(v, v)
            elif v:
                norm[i] = 255                               # wall
        out["size"] = {"x": w, "y": h}
        out["grid_rle"] = _rle(bytes(norm))
    return out


def json_room_map(data: dict) -> dict[int, int]:
    """grid_id -> room id from `map_room_info`; {} if absent or not one-to-one (ov21gl sends 0 for all)."""
    room_info = data.get("map_room_info") or []
    if not isinstance(room_info, list):
        _LOGGER.debug("xiaomi JSON grid: map_room_info is not a list")
        return {}
    grid_to_room: dict[int, int] = {}
    for entry in room_info:
        if not isinstance(entry, dict):
            continue
        try:
            grid_to_room[int(entry["grid_id"])] = int(entry["room_id"])
        except (KeyError, TypeError, ValueError):
            continue
    if len(set(grid_to_room.values())) < len(grid_to_room):
        _LOGGER.debug("xiaomi JSON grid: map_room_info is not one-to-one (%s); "
                      "using grid ids as room ids", grid_to_room)
        return {}
    return grid_to_room


def vector_map(md: Any, unpacked: bytes, *, ijai_grid: bool = True,
               json_grid: bool = False, units_per_metre: float = 1.0,
               carpets: list[list[float]] | None = None,
               path_segments: list[list[tuple[float, float]]] | None = None,
               ) -> dict[str, Any]:
    """Assemble the card contract: grid (this module) + vector overlays (md).

    `md` is the parsed vacuum_map_parser_base.MapData. All overlay coords are
    emitted in metres. `ijai_grid` is True only when `unpacked` is an ijai
    `RobotMap` protobuf; `json_grid` is True only for the xiaomi JSON-map
    family, whose decrypted payload carries its own labelled pixel grid
    (`extract_json_grid`). Both yield room contours; a brand with neither
    passes False for both and gets the overlays-only contract.

    `units_per_metre` is the divisor turning this brand's parser units into
    metres -- see `map_parsers.overlay_units_per_metre`. It is 1.0 for every
    brand that already reports metres and 1000.0 for the xiaomi JSON-map
    family, whose parser reports millimetres. Without it the card, which draws
    in metre space with absolute label and marker sizes, renders those 1000x
    too small to see.
    """
    if ijai_grid:
        out = extract_grid(unpacked)
    elif json_grid:
        out = extract_json_grid(unpacked, units_per_metre=units_per_metre)
    else:
        out = _empty_grid()

    def to_m(value: Any) -> Any:
        """Parser unit -> metre. Passes None through: `Room.pos_x`/`pos_y` are
        optional (`float | None`) and stay absent rather than becoming 0.0."""
        return value if value is None else value / units_per_metre

    if md.path is not None:
        out["path"] = [[to_m(p.x), to_m(p.y)] for sub in md.path.path for p in sub]
    if path_segments:
        out["path_segments"] = [
            [[to_m(x), to_m(y)] for x, y in segment]
            for segment in path_segments
            if len(segment) >= 2
        ]
    if carpets:
        out["carpets"] = [
            [to_m(value) for value in carpet]
            for carpet in carpets
            if len(carpet) == 8
        ]
    if md.charger is not None:
        out["charger"] = _pt(md.charger, to_m)
    if md.vacuum_position is not None:
        out["vacuum"] = _pt(md.vacuum_position, to_m)
    if md.goto is not None:
        out["goto"] = _pt(md.goto, to_m)

    out["rooms"] = [
        {
            "id": rid,
            "name": r.name,
            "cx": to_m(r.pos_x),
            "cy": to_m(r.pos_y),
            "bbox": [to_m(r.x0), to_m(r.y0), to_m(r.x1), to_m(r.y1)],
        }
        for rid, r in (md.rooms or {}).items()
    ]
    out["walls"] = [[to_m(w.x0), to_m(w.y0), to_m(w.x1), to_m(w.y1)]
                    for w in (md.walls or [])]
    out["no_go"] = [[to_m(v) for v in a.as_list()] for a in (md.no_go_areas or [])]
    out["no_mop"] = [[to_m(v) for v in a.as_list()] for a in (md.no_mopping_areas or [])]
    out["zones"] = [[to_m(z.x0), to_m(z.y0), to_m(z.x1), to_m(z.y1)]
                    for z in (md.zones or [])]
    out["vacuum_room"] = md.vacuum_room
    out["vacuum_room_name"] = md.vacuum_room_name
    return out
