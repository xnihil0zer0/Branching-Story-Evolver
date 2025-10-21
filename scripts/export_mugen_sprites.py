"""Utilities for exporting sprites and animation metadata from M.U.G.E.N characters.

This script scans a directory tree for character definition files (``.def``) and
exports the sprites from the linked ``.sff`` archives as individual PNG files
alongside JSON metadata that can later be used to rebuild sprite sheets or to
play animations.

Usage
-----
python scripts/export_mugen_sprites.py --input-root path/to/chars --output-root output/dir
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image


###############################################################################
# Basic binary helpers
###############################################################################

def _read(fmt: str, fh: io.BufferedReader) -> Tuple:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of file while reading binary data")
    return struct.unpack(fmt, data)


###############################################################################
# Palette handling
###############################################################################

class PaletteList:
    """Container that mirrors Ikemen's palette book-keeping."""

    def __init__(self) -> None:
        self.palettes: List[List[Tuple[int, int, int, int]]] = []

    def init(self) -> None:
        self.palettes = []

    def new_palette(self) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        palette: List[Tuple[int, int, int, int]] = [(0, 0, 0, 0) for _ in range(256)]
        idx = len(self.palettes)
        self.palettes.append(palette)
        return idx, palette

    def set_source(self, index: int, palette: List[Tuple[int, int, int, int]]) -> None:
        while len(self.palettes) <= index:
            self.palettes.append([(0, 0, 0, 0) for _ in range(256)])
        self.palettes[index] = palette

    def get(self, index: int) -> List[Tuple[int, int, int, int]]:
        if index < 0 or index >= len(self.palettes):
            raise IndexError(f"palette index {index} out of range")
        return self.palettes[index]


###############################################################################
# SFF parsing
###############################################################################

@dataclass
class SffHeader:
    ver0: int = 0
    ver1: int = 0
    ver2: int = 0
    ver3: int = 0
    first_sprite_header_offset: int = 0
    first_palette_header_offset: int = 0
    number_of_sprites: int = 0
    number_of_palettes: int = 0

    def read(self, fh: io.BufferedReader) -> Tuple[int, int]:
        signature = fh.read(12)
        if signature != b"ElecbyteSpr\x00":
            raise ValueError("Unrecognized SFF file, invalid header signature")
        self.ver3, self.ver2, self.ver1, self.ver0 = _read("<BBBB", fh)
        _read("<I", fh)  # skip
        lofs = tofs = 0
        if self.ver0 == 1:
            self.number_of_palettes = 0
            self.first_palette_header_offset = 0
            (self.number_of_sprites,) = _read("<I", fh)
            (self.first_sprite_header_offset,) = _read("<I", fh)
            _read("<I", fh)  # skip
        elif self.ver0 == 2:
            _read("<4I", fh)
            (self.first_sprite_header_offset,) = _read("<I", fh)
            (self.number_of_sprites,) = _read("<I", fh)
            (self.first_palette_header_offset,) = _read("<I", fh)
            (self.number_of_palettes,) = _read("<I", fh)
            (lofs,) = _read("<I", fh)
            _read("<I", fh)
            (tofs,) = _read("<I", fh)
        else:
            raise ValueError("Unrecognized SFF version")
        return lofs, tofs


@dataclass
class Sprite:
    group: int = 0
    number: int = 0
    size: Tuple[int, int] = (0, 0)
    offset: Tuple[int, int] = (0, 0)
    palette_index: int = -1
    palette: Optional[List[Tuple[int, int, int, int]]] = None
    pixel_indices: Optional[bytes] = None
    raw_pixels: Optional[bytes] = None
    raw_depth: Optional[int] = None
    rle: int = 0
    color_depth: int = 8

    def share_copy(self, other: "Sprite") -> None:
        self.size = other.size
        self.offset = other.offset
        self.palette_index = other.palette_index
        self.palette = other.palette
        self.pixel_indices = other.pixel_indices
        self.raw_pixels = other.raw_pixels
        self.raw_depth = other.raw_depth
        self.color_depth = other.color_depth
        self.rle = other.rle

    # --- decompression helpers -------------------------------------------------
    def _read_pcx_header(self, fh: io.BufferedReader, offset: int) -> None:
        fh.seek(offset)
        _read("<H", fh)  # manufacturer + version (unused)
        encoding, bpp = _read("<BB", fh)
        if bpp != 8:
            raise ValueError(f"Invalid PCX color depth: expected 8-bit, got {bpp}")
        x1, y1, x2, y2 = _read("<4H", fh)
        fh.seek(offset + 66)
        (bytes_per_line,) = _read("<H", fh)
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        self.size = (width, height)
        self.rle = bytes_per_line if encoding == 1 else 0

    def _rle_pcx_decode(self, data: bytes) -> bytes:
        if not data or self.rle <= 0:
            return data
        out = bytearray(self.size[0] * self.size[1])
        src = 0
        dst = 0
        line_pos = 0
        line_width = self.size[0]
        while dst < len(out):
            value = data[src]
            src += 1
            count = 1
            if value >= 0xC0:
                count = value & 0x3F
                value = data[src]
                src += 1
            for _ in range(count):
                if line_pos < line_width and dst < len(out):
                    out[dst] = value
                    dst += 1
                line_pos += 1
                if line_pos == self.rle:
                    line_pos = 0
                    count = 1
        self.rle = 0
        return bytes(out)

    @staticmethod
    def _rle8_decode(data: bytes, expected_size: int) -> bytes:
        if not data:
            return data
        out = bytearray(expected_size)
        src = 0
        dst = 0
        while dst < len(out) and src < len(data):
            length = 1
            value = data[src]
            src += 1
            if value & 0xC0 == 0x40:
                length = value & 0x3F
                if src >= len(data):
                    break
                value = data[src]
                src += 1
            for _ in range(length):
                if dst < len(out):
                    out[dst] = value
                    dst += 1
        return bytes(out)

    @staticmethod
    def _rle5_decode(data: bytes, expected_size: int) -> bytes:
        if not data:
            return data
        out = bytearray(expected_size)
        src = 0
        dst = 0
        while dst < len(out) and src < len(data):
            run_length = data[src]
            src += 1
            data_length = data[src] & 0x7F
            has_color = data[src] >> 7
            src += 1
            color = 0
            if has_color:
                color = data[src]
                src += 1
            while data_length >= 0:
                if dst < len(out):
                    out[dst] = color
                    dst += 1
                run_length -= 1
                if run_length < 0:
                    data_length -= 1
                    if data_length < 0 or src >= len(data):
                        break
                    color = data[src] & 0x1F
                    run_length = data[src] >> 5
                    src += 1
        return bytes(out)

    @staticmethod
    def _lz5_decode(data: bytes, expected_size: int) -> bytes:
        if not data:
            return data
        out = bytearray(expected_size)
        src = 0
        dst = 0
        ctrl = data[src]
        src += 1
        ctrl_bits = 0
        reuse_buffer = 0
        reuse_count = 0
        while dst < len(out) and src < len(data):
            if ctrl_bits >= 8:
                ctrl = data[src]
                src += 1
                ctrl_bits = 0
            token = data[src]
            src += 1
            if ctrl & (1 << ctrl_bits):
                if token & 0x3F == 0:
                    if src + 1 >= len(data):
                        break
                    token = (token << 2 | data[src]) + 1
                    src += 1
                    length = data[src] + 2
                    src += 1
                else:
                    reuse_buffer |= (token & 0xC0) >> reuse_count
                    reuse_count += 2
                    length = token & 0x3F
                    if reuse_count < 8:
                        if src >= len(data):
                            break
                        token = data[src] + 1
                        src += 1
                    else:
                        token = reuse_buffer + 1
                        reuse_buffer = 0
                        reuse_count = 0
                while length >= 0 and dst < len(out):
                    out[dst] = out[dst - token]
                    dst += 1
                    length -= 1
            else:
                if token & 0xE0 == 0:
                    if src >= len(data):
                        break
                    length = data[src] + 8
                    src += 1
                else:
                    length = token >> 5
                    token &= 0x1F
                while length > 0 and dst < len(out):
                    out[dst] = token
                    dst += 1
                    length -= 1
            ctrl_bits += 1
        return bytes(out)

    # --- data loaders ----------------------------------------------------------
    def read_header_v1(self, fh: io.BufferedReader) -> Tuple[int, int, int]:
        next_offset, data_size = _read("<II", fh)
        x_off, y_off = _read("<hh", fh)
        self.offset = (x_off, y_off)
        self.group, self.number = _read("<hh", fh)
        (link,) = _read("<H", fh)
        return next_offset, data_size, link

    def read_header_v2(
        self, fh: io.BufferedReader, lofs: int, tofs: int
    ) -> Tuple[int, int, int]:
        self.group, self.number = _read("<hh", fh)
        width, height = _read("<HH", fh)
        self.size = (width, height)
        x_off, y_off = _read("<hh", fh)
        self.offset = (x_off, y_off)
        (link,) = _read("<H", fh)
        (format_byte,) = _read("<B", fh)
        self.rle = -int(format_byte)
        (self.color_depth,) = _read("<B", fh)
        next_offset, data_size = _read("<II", fh)
        (pal_index,) = _read("<H", fh)
        self.palette_index = pal_index
        (tmp,) = _read("<H", fh)
        if tmp & 1 == 0:
            next_offset += lofs
        else:
            next_offset += tofs
        return next_offset, data_size, link

    def load_v1(
        self,
        fh: io.BufferedReader,
        data_offset: int,
        data_size: int,
        next_subheader: int,
        previous: Optional["Sprite"],
        palettes: PaletteList,
    ) -> None:
        if next_subheader > data_offset:
            data_size = next_subheader - data_offset
        fh.seek(data_offset)
        (palette_same,) = _read("<B", fh)
        palette_same = palette_same != 0 and previous is not None
        self._read_pcx_header(fh, data_offset)
        fh.seek(data_offset + 128)
        palette_size = 0 if palette_same else 768
        if data_size < 128 + palette_size:
            data_size = 128 + palette_size
        pixel_data = fh.read(data_size - (128 + palette_size))
        if palette_same:
            if previous is not None:
                self.palette_index = previous.palette_index
                self.palette = previous.palette
            if self.palette_index < 0:
                idx, pal = palettes.new_palette()
                self.palette_index = idx
                self.palette = pal
        else:
            if self.palette_index < 0:
                idx, pal = palettes.new_palette()
                self.palette_index = idx
            else:
                pal = palettes.get(self.palette_index)
            if palette_size:
                raw_palette = fh.read(palette_size)
                pal_values = list(raw_palette)
                if len(pal_values) % 3 != 0:
                    pal_values += [0] * (3 - len(pal_values) % 3)
                palette_entries = [
                    (
                        pal_values[i],
                        pal_values[i + 1],
                        pal_values[i + 2],
                        255,
                    )
                    for i in range(0, min(len(pal_values), 768), 3)
                ]
                while len(palette_entries) < 256:
                    palette_entries.append((0, 0, 0, 0))
                for idx_entry, rgba in enumerate(palette_entries):
                    r, g, b, a = rgba
                    if idx_entry == 0 and a == 255:
                        a = 0
                    pal[idx_entry] = (r, g, b, a)
            self.palette = pal
        self.pixel_indices = self._rle_pcx_decode(pixel_data)

    def load_v2(self, fh: io.BufferedReader, data_offset: int, data_size: int) -> None:
        if self.rle > 0:
            return
        elif self.rle == 0:
            fh.seek(data_offset)
            pixels = fh.read(data_size)
            if self.color_depth == 8:
                self.pixel_indices = pixels
            elif self.color_depth in (24, 32):
                self.raw_pixels = pixels
                self.raw_depth = self.color_depth
            else:
                raise ValueError("Unknown color depth")
        else:
            fh.seek(data_offset + 4)
            format_id = -self.rle
            payload = fh.read(max(0, data_size - 4)) if data_size >= 4 else b""
            decoded: Optional[bytes] = None
            if format_id == 2:
                decoded = self._rle8_decode(payload, self.size[0] * self.size[1])
            elif format_id == 3:
                decoded = self._rle5_decode(payload, self.size[0] * self.size[1])
            elif format_id == 4:
                decoded = self._lz5_decode(payload, self.size[0] * self.size[1])
            elif format_id == 10:
                image = Image.open(io.BytesIO(payload))
                if isinstance(image, Image.Image):
                    image = image.convert("RGBA")
                self.raw_pixels = image.tobytes()
                self.raw_depth = 32
                return
            elif format_id in (11, 12):
                image = Image.open(io.BytesIO(payload))
                image = image.convert("RGBA")
                self.raw_pixels = image.tobytes()
                self.raw_depth = 32
                return
            else:
                raise ValueError("Unknown SFF v2 compression format")
            self.pixel_indices = decoded


class SffArchive:
    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.header = SffHeader()
        self.sprites: Dict[Tuple[int, int], Sprite] = {}
        self.palettes = PaletteList()
        self.palettes.init()

    def load(self, *, is_character: bool = True) -> None:
        with self.filename.open("rb") as fh:
            lofs, tofs = self.header.read(fh)
            if self.header.ver0 != 1:
                unique: Dict[Tuple[int, int], int] = {}
                for index in range(self.header.number_of_palettes):
                    fh.seek(self.header.first_palette_header_offset + index * 16)
                    group, number, numcols = _read("<hhh", fh)
                    (link,) = _read("<H", fh)
                    offset, size = _read("<II", fh)
                    palette_index: int
                    palette_data: List[Tuple[int, int, int, int]]
                    key = (group, number)
                    if key in unique:
                        palette_index = unique[key]
                        palette_data = self.palettes.get(palette_index)
                    elif size == 0:
                        palette_index = int(link)
                        palette_data = self.palettes.get(palette_index)
                    else:
                        fh.seek(lofs + offset)
                        raw = fh.read(size)
                        entries: List[Tuple[int, int, int, int]] = []
                        for i in range(0, min(len(raw), 256 * 4), 4):
                            r, g, b, a = raw[i : i + 4]
                            if self.header.ver2 == 0:
                                a = 255
                            if i == 0 and a == 255:
                                a = 0
                            entries.append((r, g, b, a))
                        while len(entries) < 256:
                            entries.append((0, 0, 0, 0))
                        palette_index = index
                        palette_data = entries
                    unique[key] = palette_index
                    self.palettes.set_source(index, palette_data)
                fh.seek(0)
            prev_sprite: Optional[Sprite] = None
            subheader_offset = self.header.first_sprite_header_offset
            sprites: List[Sprite] = []
            for i in range(self.header.number_of_sprites):
                fh.seek(subheader_offset)
                sprite = Sprite()
                if self.header.ver0 == 1:
                    next_offset, data_size, link = sprite.read_header_v1(fh)
                else:
                    next_offset, data_size, link = sprite.read_header_v2(fh, lofs, tofs)
                if data_size == 0:
                    if link < len(sprites):
                        sprite.share_copy(sprites[link])
                    else:
                        sprite.palette_index = 0
                else:
                    if self.header.ver0 == 1:
                        sprite.load_v1(
                            fh,
                            subheader_offset + 32,
                            data_size,
                            next_offset,
                            prev_sprite,
                            self.palettes,
                        )
                    else:
                        sprite.load_v2(fh, next_offset, data_size)
                        if sprite.pixel_indices is not None and sprite.palette is None and sprite.palette_index >= 0:
                            try:
                                sprite.palette = self.palettes.get(sprite.palette_index)
                            except IndexError:
                                pass
                    prev_sprite = sprite
                self.sprites[(sprite.group, sprite.number)] = sprite
                sprites.append(sprite)
                if self.header.ver0 == 1:
                    subheader_offset = next_offset
                else:
                    subheader_offset += 28


###############################################################################
# AIR parsing (animation data)
###############################################################################

@dataclass
class ClsnBox:
    left: int
    top: int
    right: int
    bottom: int

    def as_list(self) -> List[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass
class AnimationFrame:
    group: int
    image: int
    x_offset: int
    y_offset: int
    duration: int
    flip: str = ""
    loopstart: bool = False
    clsn1: List[ClsnBox] = field(default_factory=list)
    clsn2: List[ClsnBox] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "group": self.group,
            "image": self.image,
            "offset": [self.x_offset, self.y_offset],
            "duration": self.duration,
            "flip": self.flip,
            "loopstart": self.loopstart,
            "clsn1": [box.as_list() for box in self.clsn1],
            "clsn2": [box.as_list() for box in self.clsn2],
        }


@dataclass
class Animation:
    action_number: int
    frames: List[AnimationFrame] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action_number,
            "frames": [frame.to_dict() for frame in self.frames],
        }


def _parse_clsn_box(text: str) -> ClsnBox:
    values = [int(v.strip()) for v in text.split(";")[0].split(",")]
    if len(values) != 4:
        raise ValueError(f"Invalid CLSN definition: {text}")
    return ClsnBox(*values)


def parse_air(path: Path) -> Dict[int, Animation]:
    animations: Dict[int, Animation] = {}
    current: Optional[Animation] = None
    default_clsn1: List[ClsnBox] = []
    default_clsn2: List[ClsnBox] = []
    pending_clsn1: Optional[List[ClsnBox]] = None
    pending_clsn2: Optional[List[ClsnBox]] = None
    pending_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("[begin action"):
                number = int(line.split()[2].strip("[]"))
                current = Animation(action_number=number)
                animations[number] = current
                pending_clsn1 = None
                pending_clsn2 = None
                default_clsn1 = []
                default_clsn2 = []
                pending_counts.clear()
                continue
            if current is None:
                continue
            if lower.startswith("clsn1default"):
                count = int(line.split(":", 1)[1])
                default_clsn1 = []
                pending_counts["clsn1default"] = count
                continue
            if lower.startswith("clsn2default"):
                count = int(line.split(":", 1)[1])
                default_clsn2 = []
                pending_counts["clsn2default"] = count
                continue
            if lower.startswith("clsn1") and "default" not in lower:
                count = int(line.split(":", 1)[1])
                pending_clsn1 = []
                pending_counts["clsn1"] = count
                continue
            if lower.startswith("clsn2") and "default" not in lower:
                count = int(line.split(":", 1)[1])
                pending_clsn2 = []
                pending_counts["clsn2"] = count
                continue
            if lower.startswith("clsn1["):
                box = _parse_clsn_box(line.split("=", 1)[1])
                if "clsn1default" in pending_counts and pending_counts["clsn1default"] > 0:
                    default_clsn1.append(box)
                    pending_counts["clsn1default"] -= 1
                elif pending_clsn1 is not None and pending_counts.get("clsn1", 0) > 0:
                    pending_clsn1.append(box)
                    pending_counts["clsn1"] -= 1
                continue
            if lower.startswith("clsn2["):
                box = _parse_clsn_box(line.split("=", 1)[1])
                if "clsn2default" in pending_counts and pending_counts["clsn2default"] > 0:
                    default_clsn2.append(box)
                    pending_counts["clsn2default"] -= 1
                elif pending_clsn2 is not None and pending_counts.get("clsn2", 0) > 0:
                    pending_clsn2.append(box)
                    pending_counts["clsn2"] -= 1
                continue
            # Frame data line
            reader = csv.reader([line], skipinitialspace=True)
            fields = [field.strip() for field in next(reader) if field.strip()]
            if len(fields) < 5:
                continue
            group = int(fields[0])
            image = int(fields[1])
            x_off = int(fields[2])
            y_off = int(fields[3])
            duration = int(fields[4])
            extras = fields[5:]
            flip = ""
            loopstart = False
            for item in extras:
                if item.lower() == "loopstart":
                    loopstart = True
                elif item.upper() in {"H", "V", "HV"}:
                    flip = item.upper()
            frame = AnimationFrame(
                group=group,
                image=image,
                x_offset=x_off,
                y_offset=y_off,
                duration=duration,
                flip=flip,
                loopstart=loopstart,
            )
            frame.clsn1 = list(pending_clsn1) if pending_clsn1 else list(default_clsn1)
            frame.clsn2 = list(pending_clsn2) if pending_clsn2 else list(default_clsn2)
            pending_clsn1 = None
            pending_clsn2 = None
            current.frames.append(frame)
    return animations


###############################################################################
# DEF parsing
###############################################################################

@dataclass
class CharacterDefinition:
    name: str
    sff_path: Path
    air_path: Optional[Path]
    root: Path


def parse_def(path: Path) -> Optional[CharacterDefinition]:
    current_section: Optional[str] = None
    info: Dict[str, str] = {}
    files: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line.strip("[]").lower()
                continue
            if "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            key_lower = key.lower()
            if current_section == "info":
                info[key_lower] = value
            elif current_section == "files":
                files[key_lower] = value
    if info.get("type", "character").lower() == "stage":
        return None
    sprite_path = files.get("sprite") or files.get("sff")
    if not sprite_path:
        return None
    air_path = files.get("anim") or files.get("air")
    root = path.parent
    sprite_file = (root / sprite_path).resolve()
    air_file = (root / air_path).resolve() if air_path else None
    name = info.get("name") or path.stem
    return CharacterDefinition(name=name, sff_path=sprite_file, air_path=air_file, root=root)


###############################################################################
# Export helpers
###############################################################################

@dataclass
class ExportOptions:
    input_root: Path
    output_root: Path
    overwrite: bool = False


def _sanitize_name(name: str) -> str:
    safe = [c if c.isalnum() or c in {"_", "-"} else "_" for c in name]
    return "".join(safe).strip("_") or "character"


def export_character(definition: CharacterDefinition, options: ExportOptions) -> None:
    archive = SffArchive(definition.sff_path)
    if not definition.sff_path.exists():
        raise FileNotFoundError(f"Sprite file not found: {definition.sff_path}")
    archive.load(is_character=True)
    sanitized = _sanitize_name(definition.name)
    character_root = options.output_root / sanitized
    sprite_root = character_root / "sprites"
    metadata_root = character_root / "metadata"
    animations_root = metadata_root / "animations"
    sprite_root.mkdir(parents=True, exist_ok=True)
    animations_root.mkdir(parents=True, exist_ok=True)

    sprite_manifest: List[Dict[str, object]] = []
    for (group, number), sprite in sorted(archive.sprites.items()):
        if sprite.raw_pixels is not None:
            mode = "RGBA" if sprite.raw_depth == 32 else "RGB"
            size = sprite.size
            image = Image.frombytes(mode, size, sprite.raw_pixels)
            if image.mode != "RGBA":
                image = image.convert("RGBA")
        else:
            if sprite.pixel_indices is None:
                continue
            palette = sprite.palette
            if palette is None and sprite.palette_index >= 0:
                try:
                    palette = archive.palettes.get(sprite.palette_index)
                except IndexError:
                    palette = None
            if palette is None:
                continue
            pixels = bytearray()
            for idx in sprite.pixel_indices:
                r, g, b, a = palette[idx]
                if idx == 0 and a == 255:
                    a = 0
                pixels.extend((r, g, b, a))
            image = Image.frombytes("RGBA", sprite.size, bytes(pixels))
        group_dir = sprite_root / f"group_{group:04d}"
        group_dir.mkdir(exist_ok=True)
        sprite_path = group_dir / f"{group:04d}_{number:04d}.png"
        if options.overwrite or not sprite_path.exists():
            image.save(sprite_path)
        sprite_manifest.append(
            {
                "group": group,
                "image": number,
                "file": str(sprite_path.relative_to(character_root)),
                "size": list(sprite.size),
                "offset": list(sprite.offset),
                "palette_index": sprite.palette_index,
            }
        )

    manifest_path = metadata_root / "sprites.json"
    manifest_path.write_text(json.dumps({"character": definition.name, "sprites": sprite_manifest}, indent=2), "utf-8")

    if definition.air_path and definition.air_path.exists():
        animations = parse_air(definition.air_path)
        for number, animation in animations.items():
            animation_path = animations_root / f"action_{number:04d}.json"
            animation_path.write_text(json.dumps(animation.to_dict(), indent=2), "utf-8")


def find_character_defs(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.def"):
        yield path


###############################################################################
# CLI
###############################################################################

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export M.U.G.E.N sprites and metadata")
    parser.add_argument("--input-root", required=True, type=Path, help="Directory containing character folders")
    parser.add_argument("--output-root", required=True, type=Path, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    args = parser.parse_args(argv)
    options = ExportOptions(input_root=args.input_root.resolve(), output_root=args.output_root.resolve(), overwrite=args.overwrite)
    options.output_root.mkdir(parents=True, exist_ok=True)

    defs_processed = 0
    for def_file in find_character_defs(options.input_root):
        definition = parse_def(def_file)
        if not definition:
            continue
        try:
            export_character(definition, options)
            defs_processed += 1
            print(f"Exported {definition.name} from {def_file}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to export {def_file}: {exc}", file=sys.stderr)
    print(f"Processed {defs_processed} character definition(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
