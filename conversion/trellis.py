"""NariQ GGUF conversion for Qwen3.5 / Qwen3.8. See nariq/FORMAT.md.

Each coded linear `<module>.{trellis, block_scale, su, sv, row_scale}` becomes a `GGML_TYPE_TRL<k>` tensor (the codes,
output-tile-major), its side tensors `<w>.trellis_{s16, pre, post}` (and `trellis_pin` / `trellis_pout` where
llama.cpp's linear-attention V-head order differs from Hugging Face's, or where a side is permuted) and the key
`trellis.desc.<w>`. A coded embedding takes the same form; a coded lm_head in frequency bands becomes
`output.band<NN>.*`, `output.vocab_gather` and `trellis.lm_head_bands`, and a coded embedding in frequency bands
(`quantization_config.embed_bands`) `token_embd.band<NN>.*`, `token_embd.vocab_gather` and `trellis.embed_bands`
(token t's row is banded row vocab_gather[t]; every band's rows unrotated, a row lookup). Every other tensor goes
through the upstream Qwen3.5 conversion unchanged.

A permuted side (`rhtp<b>`: T = H D P, (P v)[i] = v[perm[i]]; the export adds `<module>.perm_rows` / `perm_cols`
and writes that side's scales in the permuted order) is an `rht<b>` descriptor plus gathers, composed with the
V-head order: pin = perm_cols (out_proj: vhead_pin[perm_cols]), pout = argsort(perm_rows) (in_proj_*:
argsort(perm_rows)[vhead_pout]). An lm_head band's row permutation goes into `output.vocab_gather` instead.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

import gguf

from .base import LazyTorchTensor
from .qwen import Qwen3_5TextModel

logger = logging.getLogger("trellis")

PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "model.")  # upstream may rename
PARTS = ("trellis", "block_scale", "su", "sv", "row_scale")
PERMS = ("perm_rows", "perm_cols")  # a permuted side's permutation (`rhtp<b>`)
VERSION = 1


def exact_ints(t: Tensor, dtype, what: str) -> np.ndarray:
    """An integer tensor back from the float32 that upstream's prepare_tensors widens it to (exact: |x| < 2^24).
    The tensor is materialised first: the converter's tensors are lazy, and numpy on a lazy tensor sees no data."""
    a = LazyTorchTensor.to_eager(t).numpy()
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(dtype)
    info = np.iinfo(dtype)
    b = a.astype(np.int64)
    if not np.array_equal(b, a) or b.min() < info.min or b.max() > info.max:
        raise ValueError(f"{what}: not exact {np.dtype(dtype).name} values")
    return b.astype(dtype)


def side(kind: str, n: int, slice_len: int | None) -> tuple[int, int, int]:
    """(kind, block, shards) of a side record: `none`, `rht<b>` or `hadfull` (its GPU slice length)."""
    if kind == "none":
        return 0, 0, 1
    if kind.startswith("rhtp"):  # a permutation (a gather) then rht<b>
        return 1, int(kind[4:]), 1
    if kind.startswith("rht"):
        return 1, int(kind[3:]), 1
    if kind == "hadfull":
        if not slice_len or n % slice_len:
            raise ValueError(f"hadfull side of {n} with slices of {slice_len}")
        return 2, 0, n // slice_len
    raise ValueError(f"unknown side kind {kind!r}")


def descriptor(rec: dict, k: int, n: int) -> list[int]:
    """trellis.desc: [rows_kind, rows_block, rows_shards, cols_kind, cols_block, cols_shards, nparts, parts...]."""
    rows = side(rec["rows"], n, rec.get("hadamard_rows"))
    cols = side(rec["cols"], k, rec.get("hadamard_cols"))
    parts = [int(p) for p in (rec.get("row_parts") or [n])]
    if sum(parts) != n or len(parts) > 8:
        raise ValueError(f"row parts {parts} do not split {n}")
    return [*rows, *cols, len(parts), *parts]


def parts_of(rec: dict) -> tuple[str, ...]:
    """The export's tensors of a coded module: PARTS, and `perm_<side>` for each permuted side."""
    return PARTS + tuple(f"perm_{side}" for side in ("rows", "cols") if rec[side].startswith("rhtp"))


def permutation(t: Tensor, n: int, what: str) -> np.ndarray:
    p = exact_ints(t, np.int64, what)
    if p.shape != (n,) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError(f"{what}: not a permutation of {n}")
    return p


class TrellisQwen35Model(Qwen3_5TextModel):
    """The upstream Qwen3.5 text conversion, with the native export's coded tensors written as trellis tensors."""

    model_arch = gguf.MODEL_ARCH.QWEN35
    no_mtp = True  # native exports do not carry the multi-token-prediction block

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        quant = self.hparams.get("quantization_config") or {}
        if quant.get("quant_method") != "native_trellis":
            raise ValueError("not a native trellis export (quantization_config.quant_method)")
        self.trellis_records: dict = quant["modules"]
        # the banded heads: export name -> (GGUF name, record); the record is {"vocab", "bands": [{"key", "n", "record"}]}
        self.trellis_banded: dict[str, tuple[str, dict]] = {}
        if quant.get("lm_head"):
            self.trellis_banded["lm_head"] = ("output", quant["lm_head"])
        if quant.get("embed_bands"):
            self.trellis_banded["embed_tokens"] = ("token_embd", quant["embed_bands"])
        self._pending: dict[str, dict[str, Tensor]] = {}
        self._bands: dict[str, dict[int, dict[str, Tensor]]] = {h: {} for h in self.trellis_banded}
        self._vocab_order: dict[str, np.ndarray] = {}
        self._written = 0

    def dequant_model(self):
        return  # the codes are written as they are

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_uint32("trellis.version", VERSION)
        if "lm_head" in self.trellis_banded:
            self.gguf_writer.add_uint32("trellis.lm_head_bands", len(self.trellis_banded["lm_head"][1]["bands"]))
        if "embed_tokens" in self.trellis_banded:
            self.gguf_writer.add_uint32("trellis.embed_bands", len(self.trellis_banded["embed_tokens"][1]["bands"]))

    # ---- the V-head order of linear attention (upstream `_reorder_v_heads` on index vectors) ----
    def _v_perm(self, n: int, head_dim: int) -> np.ndarray:
        nk = self.hparams["linear_num_key_heads"]
        nv = self.hparams["linear_num_value_heads"]
        idx = torch.arange(n, dtype=torch.long).unsqueeze(-1)
        return self._reorder_v_heads(idx, 0, nk, nv // nk, head_dim).squeeze(-1).numpy()

    def _gathers(self, key: str, k: int, n: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(pin, pout) for a coded linear: llama.cpp's graph wants the V heads tiled, the codes stay in HF order."""
        nk = self.hparams.get("linear_num_key_heads", 0)
        nv = self.hparams.get("linear_num_value_heads", 0)
        if ".linear_attn." not in "." + key or nk <= 0 or nv <= 0 or nk == nv:
            return None, None
        hk, hv = self.hparams["linear_key_head_dim"], self.hparams["linear_value_head_dim"]
        leaf = key.rsplit(".", 1)[-1]
        if leaf == "in_proj_qkv":
            qk = 2 * hk * nk
            return None, np.concatenate([np.arange(qk), qk + self._v_perm(n - qk, hv)]).astype(np.int32)
        if leaf == "in_proj_z":
            return None, self._v_perm(n, hv).astype(np.int32)
        if leaf in ("in_proj_a", "in_proj_b"):
            return None, self._v_perm(n, 1).astype(np.int32)
        if leaf == "out_proj":  # columns: x_tiled[j] = x_hf[perm[j]], so x_hf = x_tiled[argsort(perm)]
            return np.argsort(self._v_perm(k, hv)).astype(np.int32), None
        return None, None

    # ---- writing ----
    def _write_coded(self, gguf_name: str, t: dict[str, Tensor], rec: dict, key: str | None,
                     band: bool = False) -> np.ndarray | None:
        """Write one coded module; for an lm_head band (`band`) with permuted rows, leave the rows in the permuted
        order and return argsort(perm_rows) for `output.vocab_gather`."""
        codes = exact_ints(t["trellis"], np.int16, gguf_name)
        if codes.ndim != 3 or codes.shape[2] % 16:
            raise ValueError(f"{gguf_name}: codes of shape {codes.shape} {codes.dtype}")
        kt, nt, words = codes.shape
        bits, k, n = words // 16, 16 * kt, 16 * nt
        if bits != int(rec["bits"]):
            raise ValueError(f"{gguf_name}: {bits}-bit codes, record says {rec['bits']}")
        raw = np.ascontiguousarray(codes.transpose(1, 0, 2)).view(np.uint8).reshape(n, k * bits // 8)
        qtype = getattr(gguf.GGMLQuantizationType, f"TRL{bits}")
        base = gguf_name[: -len(".weight")]
        self.gguf_writer.add_tensor(gguf_name, raw, raw_dtype=qtype)
        eager = {name: LazyTorchTensor.to_eager(v) for name, v in t.items() if name not in ("trellis", *PERMS)}
        self.gguf_writer.add_tensor(base + ".trellis_s16", eager["block_scale"].to(torch.float16).numpy())
        self.gguf_writer.add_tensor(base + ".trellis_pre", eager["su"].float().numpy())
        self.gguf_writer.add_tensor(base + ".trellis_post", (eager["sv"].float() * eager["row_scale"].float()).numpy())
        pin, pout = self._gathers(key, k, n) if key else (None, None)
        inverse_rows = None
        if "perm_cols" in t:  # x_perm[i] = x_hf[perm[i]]; out_proj's x_hf = x_tiled[vhead_pin]
            perm = permutation(t["perm_cols"], k, gguf_name + " perm_cols")
            pin = (perm if pin is None else pin.astype(np.int64)[perm]).astype(np.int32)
        if "perm_rows" in t:  # y_hf[j] = y_perm[argsort(perm)[j]]; in_proj_*'s y_tiled[j] = y_hf[vhead_pout[j]]
            inverse_rows = np.argsort(permutation(t["perm_rows"], n, gguf_name + " perm_rows"))
            if band:
                if pout is not None:
                    raise ValueError(f"{gguf_name}: a band with an output gather")
            else:
                pout = (inverse_rows if pout is None else inverse_rows[pout]).astype(np.int32)
        if pin is not None:
            self.gguf_writer.add_tensor(base + ".trellis_pin", pin)
        if pout is not None:
            self.gguf_writer.add_tensor(base + ".trellis_pout", pout)
        self.gguf_writer.add_array(f"trellis.desc.{base}", descriptor(rec, k, n))
        self._written += 1
        logger.info(f"{gguf_name:<40} trellis {bits} bits, {{{k}, {n}}}, rows {rec['rows']}, cols {rec['cols']}"
                    + (", input gather" if pin is not None else "") + (", output gather" if pout is not None else "")
                    + (", rows into the vocab gather" if band and inverse_rows is not None else ""))
        return inverse_rows if band else None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        module, _, leaf = name.rpartition(".")
        key = self._record_key(module)
        if leaf in PARTS + PERMS and key is not None:
            parts = self._pending.setdefault(key, {})
            parts[leaf] = data_torch
            if set(parts) == set(parts_of(self.trellis_records[key])):
                self._write_coded(self.map_tensor_name(module + ".weight"), parts, self.trellis_records[key], key)
                del self._pending[key]
            return
        banded = self._banded_part(name)
        if banded is not None:
            self._collect_band(*banded, data_torch)
            return
        if "lm_head" in self.trellis_banded and name == "lm_head.weight":
            return  # the BF16 copy: the coded bands replace it
        yield from super().modify_tensors(data_torch, name, bid)

    def _banded_part(self, name: str) -> tuple[str, str] | None:
        """(head, rest) of a banded head's tensor: `lm_head.band<NN>_<part>`, `lm_head.vocab_order` and the same under
        `<prefix>embed_tokens` (rest `band<NN>_<part>` or `vocab_order`); None for any other tensor."""
        for head in self.trellis_banded:
            for prefix in (("",) if head == "lm_head" else PREFIXES):
                stem = prefix + head + "."
                if name.startswith(stem):
                    rest = name[len(stem):]
                    if rest == "vocab_order" or (rest.startswith("band") and "_" in rest):
                        return head, rest
        return None

    def _record_key(self, module: str) -> str | None:
        """The quantization_config record of a module name (records are keyed without the model prefix)."""
        for prefix in PREFIXES + ("",):
            if module.startswith(prefix) and module[len(prefix):] in self.trellis_records:
                return module[len(prefix):]
        return None

    def _collect_band(self, head: str, rest: str, data: Tensor):
        """A banded head's `band<NN>_<part>` or `vocab_order` (native/convert.py, lm_head_bands; the llama.cpp export's
        embedding bands likewise)."""
        if rest == "vocab_order":
            self._vocab_order[head] = exact_ints(data, np.int64, f"{head}.vocab_order")
        else:
            band, _, part = rest[len("band"):].partition("_")
            self._bands[head].setdefault(int(band), {})[part] = data
        self._flush_bands(head)

    def _flush_bands(self, head: str):
        gguf_name, rec_all = self.trellis_banded[head]
        bands = self._bands[head]
        if head not in self._vocab_order or len(bands) != len(rec_all["bands"]) or \
                any(set(p) != set(parts_of(rec_all["bands"][b]["record"])) for b, p in bands.items()):
            return
        order = self._vocab_order[head]
        position = np.arange(order.shape[0])  # banded position of band order's position (permuted bands)
        start = 0
        for b in sorted(bands):
            rec = rec_all["bands"][b]["record"]  # {"key", "n", "record"} per band (native/convert.py, lm_head_bands)
            if head == "embed_tokens" and rec["rows"] != "none":
                raise ValueError(f"trellis: embedding band {b} has rows {rec['rows']}; a row lookup needs none")
            inverse = self._write_coded(f"{gguf_name}.band{b:02d}.weight", bands[b], rec, None, band=True)
            n = int(rec_all["bands"][b]["n"])
            if inverse is not None:  # the band's row j (vocabulary order[start + j]) is banded row start + inverse[j]
                position[start:start + n] = start + inverse
            start += n
        if start != order.shape[0]:
            raise ValueError(f"trellis: the {head} bands hold {start} rows, the vocabulary {order.shape[0]}")
        gather = np.empty_like(order)
        gather[order] = position  # lm_head: logits[t] = banded[gather[t]]; embedding: row t = banded row gather[t]
        self.gguf_writer.add_tensor(f"{gguf_name}.vocab_gather", gather.astype(np.int32))
        bands.clear()
        del self._vocab_order[head]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._pending:
            raise ValueError(f"trellis: incomplete coded modules {sorted(self._pending)[:4]}")
        if any(self._bands.values()) or self._vocab_order:
            raise ValueError(f"trellis: incomplete bands of {sorted(h for h, b in self._bands.items() if b)}")
        logger.info(f"trellis: {self._written} coded tensors")
