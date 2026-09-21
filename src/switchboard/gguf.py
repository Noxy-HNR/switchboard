"""Minimal GGUF metadata reader.

Only the header is read, never the tensor data, so this stays fast on a 16 GB
file. It answers three questions the router needs and cannot get anywhere else:

* the model's native context length (the configured context may be lower);
* how many layers it has, so the host-RAM share of a partial offload can be
  estimated from the configured ``gpuLayers``;
* whether its chat template can express tool calls at all.

Format reference: ggml-org/llama.cpp docs/gguf.md.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_MAGIC = b"GGUF"

# GGUF metadata value types.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED_WIDTH = {
    _UINT8: 1, _INT8: 1, _BOOL: 1,
    _UINT16: 2, _INT16: 2,
    _UINT32: 4, _INT32: 4, _FLOAT32: 4,
    _UINT64: 8, _INT64: 8, _FLOAT64: 8,
}
_FIXED_FORMAT = {
    _UINT8: "<B", _INT8: "<b", _BOOL: "<?",
    _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f",
    _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}

# Strings longer than this are metadata we do not need (vocabularies, merges).
_MAX_STRING = 1 << 20


@dataclass
class GgufInfo:
    """What the header told us. Every field may be None on an unreadable file."""

    architecture: str | None = None
    context_length: int | None = None
    block_count: int | None = None
    chat_template: str | None = None
    name: str | None = None
    error: str | None = None

    @property
    def supports_tools(self) -> bool | None:
        """Whether the chat template references tools.

        llama.cpp renders tool definitions through the model's Jinja template. A
        template that never mentions ``tools`` silently drops them, so a request
        carrying tool definitions should not be routed to that model.
        """
        if self.chat_template is None:
            return None
        return "tools" in self.chat_template


def _read(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return data


def _read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", _read(stream, 4))[0]


def _read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", _read(stream, 8))[0]


def _read_string(stream: BinaryIO) -> str:
    length = _read_u64(stream)
    if length > _MAX_STRING:
        stream.seek(length, 1)
        return ""
    return _read(stream, length).decode("utf-8", "replace")


def _skip_value(stream: BinaryIO, value_type: int) -> None:
    """Advance past a value without materialising it."""
    if value_type in _FIXED_WIDTH:
        stream.seek(_FIXED_WIDTH[value_type], 1)
    elif value_type == _STRING:
        stream.seek(_read_u64(stream), 1)
    elif value_type == _ARRAY:
        item_type = _read_u32(stream)
        count = _read_u64(stream)
        if item_type in _FIXED_WIDTH:
            stream.seek(_FIXED_WIDTH[item_type] * count, 1)
        elif item_type == _STRING:
            # A vocabulary array: lengths must be walked one by one.
            for _ in range(count):
                stream.seek(_read_u64(stream), 1)
        else:
            raise ValueError(f"unsupported GGUF array item type {item_type}")
    else:
        raise ValueError(f"unsupported GGUF value type {value_type}")


def _read_value(stream: BinaryIO, value_type: int) -> object:
    if value_type in _FIXED_FORMAT:
        return struct.unpack(_FIXED_FORMAT[value_type], _read(stream, _FIXED_WIDTH[value_type]))[0]
    if value_type == _STRING:
        return _read_string(stream)
    _skip_value(stream, value_type)
    return None


def read_gguf_info(path: Path) -> GgufInfo:
    """Read the interesting metadata keys out of a GGUF file."""
    try:
        with path.open("rb") as stream:
            if _read(stream, 4) != _MAGIC:
                return GgufInfo(error="not a GGUF file")
            _read_u32(stream)  # format version
            _read_u64(stream)  # tensor count
            kv_count = _read_u64(stream)

            raw: dict[str, object] = {}
            wanted_suffixes = (".context_length", ".block_count")
            wanted_exact = {"general.architecture", "general.name", "tokenizer.chat_template"}

            for _ in range(kv_count):
                key = _read_string(stream)
                value_type = _read_u32(stream)
                if key in wanted_exact or key.endswith(wanted_suffixes):
                    raw[key] = _read_value(stream, value_type)
                else:
                    _skip_value(stream, value_type)

            architecture = raw.get("general.architecture")
            arch = architecture if isinstance(architecture, str) else None

            def _int_for(suffix: str) -> int | None:
                if arch and isinstance(value := raw.get(f"{arch}{suffix}"), int):
                    return value
                # Fall back to any key with the suffix: some converters disagree
                # with general.architecture about the prefix.
                for key, value in raw.items():
                    if key.endswith(suffix) and isinstance(value, int):
                        return value
                return None

            template = raw.get("tokenizer.chat_template")
            name = raw.get("general.name")
            return GgufInfo(
                architecture=arch,
                context_length=_int_for(".context_length"),
                block_count=_int_for(".block_count"),
                chat_template=template if isinstance(template, str) else None,
                name=name if isinstance(name, str) else None,
            )
    except (OSError, ValueError, struct.error) as error:
        return GgufInfo(error=f"{type(error).__name__}: {error}")
