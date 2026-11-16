"""Block-wise quantization for persisted Adam state.

The storage path keeps an encoded :class:`StatePayload` opaque until a block is
admitted to the GPU.  The INT8 map follows the dynamic map used by the ICLR
2022 8-bit optimizer work: one global map, an independent absolute-max scale
for each tile, and separate signed/positive maps for the two Adam moments.
"""

from __future__ import annotations

import struct
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch


STATE_WIDTH = 118
DEFAULT_QTILE_SIZE = 512
CODEBOOK_ID = "dynamic_signed_positive_v1"
SIGNED_CODEBOOK_ID = "dynamic_signed_v1"
POSITIVE_CODEBOOK_ID = "dynamic_positive_v1"
SIGNED_CODEBOOK_TAG = 1
POSITIVE_CODEBOOK_TAG = 2

MAGIC = b"TIDEQST1"
PAYLOAD_VERSION = 1

# Header is deliberately fixed-width and little-endian.  The two codebook IDs
# are numeric tags; their textual names are stable module constants above.
HEADER = struct.Struct("<8sHHIHHIIIIHHII")
# kind, mode, flags, logical offset, element count, nonzero count,
# data offset, data size, scale, reserved
DESCRIPTOR = struct.Struct("<BBHIIIIIfI")

KIND_M = 0
KIND_V = 1

MODE_ZERO = 0
MODE_DENSE_INT8 = 1
MODE_SPARSE_INT8 = 2
MODE_FP16 = 3
MODE_FP32 = 4

CODEC_FP32 = "fp32_v1"
CODEC_FP16 = "fp16_v1"
CODEC_INT8 = "dynamic_int8_v1"

# Fallback thresholds are intentionally conservative for Adam's second moment.
MAX_M_NORMALIZED_ABS_ERROR = 0.125
MAX_V_RELATIVE_ERROR = 0.25

_MODE_NAMES = {
    MODE_ZERO: "zero",
    MODE_DENSE_INT8: "dense_int8",
    MODE_SPARSE_INT8: "sparse_int8",
    MODE_FP16: "fp16",
    MODE_FP32: "fp32",
}


@dataclass(frozen=True)
class StatePayload:
    """Opaque state bytes carried by the RAM cache and SSD extent.

    ``payload`` is always CPU-owned bytes.  It contains the fixed header and
    tile descriptors for encoded formats, while ``raw_size`` remains the
    uncompressed FP32 size used for validation and metrics.
    """

    codec: str
    row_count: int
    raw_size: int
    payload: bytes
    qtile_size: int = 0
    codebook_id: str = ""
    zero_tiles: int = 0
    sparse_tiles: int = 0
    fp16_fallback_tiles: int = 0
    fp32_fallback_tiles: int = 0

    @property
    def compressed_size(self) -> int:
        return len(self.payload)

    @property
    def is_compressed(self) -> bool:
        return self.codec != CODEC_FP32

    def as_bytes(self) -> bytes:
        return bytes(self.payload)


def _create_dynamic_map(signed: bool) -> torch.Tensor:
    """Reproduce bitsandbytes' deterministic dynamic map construction."""

    total_bits = 8
    max_exponent_bits = 7
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    data = []
    for exponent in range(max_exponent_bits):
        fraction_items = int(
            2 ** (exponent + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (exponent + non_sign_bits - max_exponent_bits + 1) + 1
        )
        boundaries = torch.linspace(0.1, 1.0, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        magnitude = 10.0 ** (-(max_exponent_bits - 1) + exponent)
        data.extend((magnitude * means).tolist())
        if signed:
            data.extend((-magnitude * means).tolist())
    if additional_items > 0:
        boundaries = torch.linspace(
            0.1, 1.0, additional_items + 1, dtype=torch.float32
        )
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        magnitude = 10.0 ** (-(max_exponent_bits - 1) + exponent)
        data.extend((magnitude * means).tolist())
        if signed:
            data.extend((-magnitude * means).tolist())
    data.extend([0.0, 1.0])
    if len(data) != 2**total_bits:
        raise RuntimeError(f"dynamic map has {len(data)} entries, expected 256")
    return torch.tensor(sorted(data), dtype=torch.float32)


_SIGNED_CODEBOOK = _create_dynamic_map(True)
_POSITIVE_CODEBOOK = _create_dynamic_map(False)


def get_codebook(kind: int, device: Optional[torch.device] = None) -> torch.Tensor:
    if kind == KIND_M:
        codebook = _SIGNED_CODEBOOK
    elif kind == KIND_V:
        codebook = _POSITIVE_CODEBOOK
    else:
        raise ValueError(f"unknown state kind {kind}")
    return codebook if device is None else codebook.to(device=device)


def codebook_ids() -> tuple[str, str]:
    return SIGNED_CODEBOOK_ID, POSITIVE_CODEBOOK_ID


def _nearest_code(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest sorted codebook entry without materialising a 256-way diff."""

    clipped = values.clamp(float(codebook[0].item()), float(codebook[-1].item()))
    right = torch.searchsorted(codebook, clipped, right=False).clamp(0, 255)
    left = (right - 1).clamp(0, 255)
    right_error = (codebook[right] - clipped).abs()
    left_error = (codebook[left] - clipped).abs()
    return torch.where(left_error <= right_error, left, right).to(torch.uint8)


def _error_ok(kind: int, original: torch.Tensor, decoded: torch.Tensor, scale: float) -> bool:
    if not bool(torch.isfinite(decoded).all().item()):
        return False
    if kind == KIND_M:
        if original.numel() == 0:
            return True
        error = (decoded - original).abs().max().item()
        return error <= MAX_M_NORMALIZED_ABS_ERROR * max(scale, 0.0)
    nonzero = original != 0
    if not bool(nonzero.any().item()):
        return True
    denom = original[nonzero].abs().clamp_min(1.0e-12)
    relative = ((decoded[nonzero] - original[nonzero]).abs() / denom).max().item()
    return relative <= MAX_V_RELATIVE_ERROR


def _bitmap(mask: torch.Tensor) -> bytes:
    values = mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
    packed = np.packbits(values, bitorder="little")
    return packed.tobytes()


def _unpack_bitmap(data: bytes, count: int) -> np.ndarray:
    if len(data) != (count + 7) // 8:
        raise ValueError("invalid sparse state bitmap length")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[:count].astype(bool)


def _raw_payload(flat_state: torch.Tensor, codec: str, qtile_size: int) -> StatePayload:
    row_count = int(flat_state.shape[0])
    raw_size = int(flat_state.numel() * 4)
    if codec == CODEC_FP16:
        data = flat_state.detach().to(dtype=torch.float16, device="cpu").numpy().tobytes()
    elif codec == CODEC_FP32:
        data = flat_state.detach().to(dtype=torch.float32, device="cpu").numpy().tobytes()
    else:
        raise ValueError(f"unsupported raw state codec {codec!r}")
    return StatePayload(codec, row_count, raw_size, data, qtile_size=qtile_size)


def _iter_groups(row_count: int) -> Iterable[tuple[int, int, int]]:
    widths = (3, 1, 3, 4, 3, 45)
    for kind in (KIND_M, KIND_V):
        base = kind * 59
        cursor = base
        for group_id, width in enumerate(widths):
            yield kind, group_id, cursor
            cursor += width


def encode_state(
    flat_state: torch.Tensor,
    *,
    quantization: str = "int8_dynamic",
    qtile_size: int = DEFAULT_QTILE_SIZE,
) -> StatePayload:
    """Encode an ``(rows, 118)`` FP32 state tensor.

    Quantization and error checks happen on the source device.  Only encoded
    bytes are copied to CPU, so a CUDA caller does not perform a FP32 D2H copy.
    """

    if not torch.is_tensor(flat_state):
        flat_state = torch.as_tensor(flat_state, dtype=torch.float32)
    if flat_state.dim() != 2 or int(flat_state.shape[1]) != STATE_WIDTH:
        raise ValueError(
            f"state must have shape (rows, {STATE_WIDTH}), got {tuple(flat_state.shape)}"
        )
    qtile_size = int(qtile_size)
    if qtile_size <= 0 or qtile_size > 0xFFFF:
        raise ValueError("qtile_size must be in [1, 65535]")
    flat_state = flat_state.detach().to(dtype=torch.float32).contiguous()
    quantization = str(quantization).lower()
    if quantization == "off":
        return _raw_payload(flat_state, CODEC_FP32, qtile_size)
    if quantization == "fp16":
        return _raw_payload(flat_state, CODEC_FP16, qtile_size)
    if quantization != "int8_dynamic":
        raise ValueError(
            f"unsupported state quantization {quantization!r}; expected off, fp16, int8_dynamic"
        )

    row_count = int(flat_state.shape[0])
    raw_size = int(flat_state.numel() * 4)
    descriptors: list[tuple[int, int, int, int, int, int, int, int, float]] = []
    data_parts: list[bytes] = []
    data_cursor = 0
    zero_tiles = 0
    sparse_tiles = 0
    fp16_fallback_tiles = 0
    fp32_fallback_tiles = 0

    for kind, group_id, logical_base in _iter_groups(row_count):
        width = 0
        # The next group boundary is obtained from the fixed parameter layout.
        for candidate_kind, candidate_group, candidate_base in _iter_groups(row_count):
            if candidate_kind == kind and candidate_base == logical_base:
                width = (3, 1, 3, 4, 3, 45)[candidate_group]
                break
        if width <= 0:
            raise RuntimeError("failed to resolve state component width")
        column = logical_base - kind * 59
        values = flat_state[:, kind * 59 + column : kind * 59 + column + width].reshape(-1)
        for offset in range(0, int(values.numel()), qtile_size):
            tile = values[offset : offset + qtile_size]
            count = int(tile.numel())
            finite = bool(torch.isfinite(tile).all().item())
            nonzero = tile != 0
            nnz = int(nonzero.sum().item())
            # The descriptor offset is relative to this component group.  A
            # row-major ``(rows, 118)`` tensor is not contiguous by component,
            # so the decoder uses ``kind`` and ``group_id`` to scatter tiles
            # back into their original columns.
            logical_offset = int(offset)
            if nnz == 0 and finite:
                zero_tiles += 1
                descriptors.append((kind, group_id, MODE_ZERO, logical_offset, count, 0, data_cursor, 0, 0.0))
                continue

            scale_tensor = tile.abs().max()
            scale = float(scale_tensor.detach().cpu().item())
            invalid_positive = kind == KIND_V and bool((tile[nonzero] < 0).any().item())
            use_int8 = finite and not invalid_positive and scale > 0.0
            mode = MODE_DENSE_INT8
            encoded = b""
            if use_int8:
                codebook = get_codebook(kind, flat_state.device)
                normalized = (tile / scale).clamp(float(codebook[0].item()), float(codebook[-1].item()))
                codes = _nearest_code(normalized, codebook)
                decoded = codebook[codes.long()] * scale
                if not _error_ok(kind, tile, decoded, scale):
                    use_int8 = False
                else:
                    dense = codes.detach().to(device="cpu", dtype=torch.uint8).numpy().tobytes()
                    sparse = _bitmap(nonzero) + codes[nonzero].detach().to(device="cpu", dtype=torch.uint8).numpy().tobytes()
                    if len(sparse) < len(dense):
                        mode = MODE_SPARSE_INT8
                        sparse_tiles += 1
                        encoded = sparse
                    else:
                        encoded = dense

            if not use_int8:
                fp16_ok = finite and bool(torch.isfinite(tile.to(torch.float16)).all().item())
                if fp16_ok:
                    fp16_decoded = tile.to(torch.float16).to(torch.float32)
                    fp16_ok = _error_ok(kind, tile, fp16_decoded, scale)
                if fp16_ok:
                    mode = MODE_FP16
                    fp16_fallback_tiles += 1
                    encoded = tile.detach().to(device="cpu", dtype=torch.float16).numpy().tobytes()
                else:
                    mode = MODE_FP32
                    fp32_fallback_tiles += 1
                    encoded = tile.detach().to(device="cpu", dtype=torch.float32).numpy().tobytes()

            data_parts.append(encoded)
            descriptor_scale = scale if math.isfinite(scale) else 0.0
            descriptors.append(
                (
                    kind,
                    group_id,
                    mode,
                    logical_offset,
                    count,
                    nnz,
                    data_cursor,
                    len(encoded),
                    descriptor_scale,
                )
            )
            data_cursor += len(encoded)

    tile_count = len(descriptors)
    header_size = HEADER.size + tile_count * DESCRIPTOR.size
    header = HEADER.pack(
        MAGIC,
        PAYLOAD_VERSION,
        0,
        row_count,
        STATE_WIDTH,
        qtile_size,
        tile_count,
        raw_size,
        header_size,
        data_cursor,
        SIGNED_CODEBOOK_TAG,
        POSITIVE_CODEBOOK_TAG,
        0,
        0,
    )
    descriptor_bytes = bytearray()
    for kind, group_id, mode, logical_offset, count, nnz, data_offset, data_size, scale in descriptors:
        descriptor_bytes.extend(
            DESCRIPTOR.pack(
                int(kind),
                int(mode),
                int(group_id),
                int(logical_offset),
                int(count),
                int(nnz),
                int(data_offset),
                int(data_size),
                float(scale),
                0,
            )
        )
    payload = bytes(header + descriptor_bytes + b"".join(data_parts))
    return StatePayload(
        CODEC_INT8,
        row_count,
        raw_size,
        payload,
        qtile_size=qtile_size,
        codebook_id=CODEBOOK_ID,
        zero_tiles=zero_tiles,
        sparse_tiles=sparse_tiles,
        fp16_fallback_tiles=fp16_fallback_tiles,
        fp32_fallback_tiles=fp32_fallback_tiles,
    )


def _parse_payload(payload: StatePayload):
    if payload.codec != CODEC_INT8:
        return None
    data = payload.payload
    if len(data) < HEADER.size:
        raise ValueError("state payload is shorter than its header")
    (
        magic,
        version,
        _flags,
        row_count,
        width,
        qtile_size,
        tile_count,
        raw_size,
        header_size,
        payload_data_size,
        m_codebook,
        v_codebook,
        _reserved0,
        _reserved1,
    ) = HEADER.unpack_from(data, 0)
    if magic != MAGIC or int(version) != PAYLOAD_VERSION:
        raise ValueError("unsupported state payload magic/version")
    if int(m_codebook) != SIGNED_CODEBOOK_TAG or int(v_codebook) != POSITIVE_CODEBOOK_TAG:
        raise ValueError("state payload uses an unsupported codebook")
    if payload.codebook_id and payload.codebook_id != CODEBOOK_ID:
        raise ValueError("state payload codebook metadata mismatch")
    if qtile_size <= 0 or tile_count <= 0:
        raise ValueError("state payload has invalid tile metadata")
    if payload.qtile_size and int(payload.qtile_size) != int(qtile_size):
        raise ValueError("state payload tile-size metadata mismatch")
    if width != STATE_WIDTH or row_count != payload.row_count:
        raise ValueError("state payload shape metadata mismatch")
    if raw_size != payload.raw_size:
        raise ValueError("state payload raw-size metadata mismatch")
    if header_size != HEADER.size + tile_count * DESCRIPTOR.size:
        raise ValueError("state payload header size mismatch")
    if header_size + payload_data_size != len(data):
        raise ValueError("state payload data size mismatch")
    descriptors = []
    group_ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index in range(tile_count):
        try:
            descriptor = DESCRIPTOR.unpack_from(
                data, HEADER.size + index * DESCRIPTOR.size
            )
        except struct.error as error:
            raise ValueError("state payload descriptor table is truncated") from error
        kind, mode, group_id, logical_offset, count, nonzero_count, data_offset, data_size, scale, _reserved = descriptor
        if kind not in (KIND_M, KIND_V):
            raise ValueError("state payload has an invalid state kind")
        if mode not in _MODE_NAMES:
            raise ValueError("state payload has an invalid tile mode")
        widths = (3, 1, 3, 4, 3, 45)
        if group_id >= len(widths) or count <= 0:
            raise ValueError("state payload has an invalid tile descriptor")
        if logical_offset + count > row_count * widths[group_id]:
            raise ValueError("state payload tile exceeds its component group")
        group_ranges.setdefault((int(kind), int(group_id)), []).append(
            (int(logical_offset), int(logical_offset + count))
        )
        if data_offset + data_size > payload_data_size:
            raise ValueError("state payload tile exceeds its data section")
        if not np.isfinite(scale):
            raise ValueError("state payload tile scale is not finite")
        if nonzero_count > count:
            raise ValueError("state payload has an invalid nonzero count")
        if mode == MODE_ZERO and (data_size != 0 or nonzero_count != 0):
            raise ValueError("zero state tile must not contain data")
        if mode in (MODE_DENSE_INT8, MODE_SPARSE_INT8) and scale <= 0.0:
            raise ValueError("INT8 state tile scale must be positive")
        if mode == MODE_DENSE_INT8 and data_size != count:
            raise ValueError("dense INT8 tile length mismatch")
        if mode == MODE_SPARSE_INT8:
            if data_size != (count + 7) // 8 + nonzero_count:
                raise ValueError("sparse INT8 tile length mismatch")
        if mode == MODE_FP16 and data_size != count * 2:
            raise ValueError("FP16 tile length mismatch")
        if mode == MODE_FP32 and data_size != count * 4:
            raise ValueError("FP32 tile length mismatch")
        descriptors.append(descriptor)
    for kind in (KIND_M, KIND_V):
        for group_id, width in enumerate((3, 1, 3, 4, 3, 45)):
            ranges = sorted(group_ranges.get((kind, group_id), []))
            cursor = 0
            for start, end in ranges:
                if start != cursor:
                    raise ValueError("state payload has a gap or overlap between tiles")
                cursor = end
            if cursor != row_count * width:
                raise ValueError("state payload does not cover every state scalar")
    return descriptors, header_size


def validate_state_payload(payload: StatePayload) -> None:
    if not isinstance(payload, StatePayload):
        raise TypeError(f"expected StatePayload, got {type(payload).__name__}")
    if payload.row_count <= 0 or payload.raw_size != payload.row_count * STATE_WIDTH * 4:
        raise ValueError("invalid state payload row_count/raw_size")
    if payload.codec == CODEC_INT8:
        _parse_payload(payload)
    elif payload.codec == CODEC_FP16:
        if len(payload.payload) != payload.row_count * STATE_WIDTH * 2:
            raise ValueError("invalid FP16 state payload length")
    elif payload.codec == CODEC_FP32:
        if len(payload.payload) != payload.raw_size:
            raise ValueError("invalid FP32 state payload length")
    else:
        raise ValueError(f"unsupported state payload codec {payload.codec!r}")


def decode_state(payload: StatePayload | torch.Tensor, device: Optional[torch.device | str] = None) -> torch.Tensor:
    """Decode a state payload to an FP32 ``(rows, 118)`` tensor."""

    if torch.is_tensor(payload):
        result = payload.to(dtype=torch.float32, device=device or payload.device)
        return result.reshape(-1, STATE_WIDTH).contiguous()
    validate_state_payload(payload)
    target = torch.device(device) if device is not None else torch.device("cpu")
    if payload.codec == CODEC_FP32:
        array = np.frombuffer(payload.payload, dtype=np.float32).copy()
        return torch.from_numpy(array).reshape(payload.row_count, STATE_WIDTH).to(target)
    if payload.codec == CODEC_FP16:
        array = np.frombuffer(payload.payload, dtype=np.float16).copy()
        return torch.from_numpy(array).reshape(payload.row_count, STATE_WIDTH).to(target, dtype=torch.float32)

    descriptors, data_start = _parse_payload(payload)
    output = torch.zeros(payload.row_count * STATE_WIDTH, dtype=torch.float32, device=target)
    host = payload.payload
    for descriptor in descriptors:
        kind, mode, group_id, logical_offset, count, nonzero_count, data_offset, data_size, scale, _reserved = descriptor
        start = data_start + int(data_offset)
        end = start + int(data_size)
        raw = host[start:end]
        if mode == MODE_ZERO:
            continue
        if mode == MODE_DENSE_INT8:
            if len(raw) != count:
                raise ValueError("dense INT8 tile length mismatch")
            codes = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(target, dtype=torch.long)
            values = get_codebook(kind, target)[codes] * float(scale)
        elif mode == MODE_SPARSE_INT8:
            bitmap_size = (count + 7) // 8
            if len(raw) != bitmap_size + nonzero_count:
                raise ValueError("sparse INT8 tile length mismatch")
            mask = torch.from_numpy(_unpack_bitmap(raw[:bitmap_size], count)).to(target)
            codes = torch.from_numpy(np.frombuffer(raw[bitmap_size:], dtype=np.uint8).copy()).to(target, dtype=torch.long)
            values = torch.zeros(count, dtype=torch.float32, device=target)
            values[mask] = get_codebook(kind, target)[codes] * float(scale)
        elif mode == MODE_FP16:
            if len(raw) != count * 2:
                raise ValueError("FP16 tile length mismatch")
            values = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).to(target, dtype=torch.float32)
        elif mode == MODE_FP32:
            if len(raw) != count * 4:
                raise ValueError("FP32 tile length mismatch")
            values = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).to(target)
        else:
            raise ValueError(f"unknown state tile mode {mode}")
        widths = (3, 1, 3, 4, 3, 45)
        if int(group_id) < 0 or int(group_id) >= len(widths):
            raise ValueError("state tile has invalid component group")
        component_start = int(kind) * 59 + sum(widths[: int(group_id)])
        component_width = widths[int(group_id)]
        if int(logical_offset) + int(count) > payload.row_count * component_width:
            raise ValueError("state tile exceeds component group")
        linear = torch.arange(
            int(logical_offset),
            int(logical_offset) + int(count),
            device=target,
            dtype=torch.long,
        )
        row_indices = torch.div(linear, component_width, rounding_mode="floor")
        column_indices = linear.remainder(component_width) + component_start
        output_view = output.reshape(payload.row_count, STATE_WIDTH)
        output_view[row_indices, column_indices] = values
    return output.reshape(payload.row_count, STATE_WIDTH)


def state_payload_from_tensor(tensor: torch.Tensor, *, codec: str = CODEC_FP32, qtile_size: int = DEFAULT_QTILE_SIZE) -> StatePayload:
    """Wrap a legacy FP32 tensor or encode it using the requested codec."""

    if codec == CODEC_INT8:
        return encode_state(tensor, quantization="int8_dynamic", qtile_size=qtile_size)
    if codec == CODEC_FP16:
        return encode_state(tensor, quantization="fp16", qtile_size=qtile_size)
    if codec == CODEC_FP32:
        return encode_state(tensor, quantization="off", qtile_size=qtile_size)
    raise ValueError(f"unsupported state codec {codec!r}")


__all__ = [
    "CODEBOOK_ID",
    "SIGNED_CODEBOOK_ID",
    "POSITIVE_CODEBOOK_ID",
    "CODEC_FP16",
    "CODEC_FP32",
    "CODEC_INT8",
    "DEFAULT_QTILE_SIZE",
    "StatePayload",
    "decode_state",
    "encode_state",
    "codebook_ids",
    "get_codebook",
    "state_payload_from_tensor",
    "validate_state_payload",
]
