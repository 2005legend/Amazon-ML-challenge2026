"""Indic-script → Latin maps learned only from training pairs, with an anyascii fallback."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from anyascii import anyascii

from .normalize import INDIC_RE, LEGAL, STATE_MAPS, _PUNCT, _WS, clean_tokens, fold


def comp_key(text: str) -> str:
    """Same component cleaning that normalize_address applies before calling translit."""
    return _WS.sub(" ", _PUNCT.sub(" ", fold(text).replace(".", ""))).strip()


def _select(counts: dict[str, Counter], min_count: int, min_share: float) -> dict[str, str]:
    out = {}
    for key, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            out[key] = best
    return out


def learn_token_map(pairs: Iterable[tuple[str, str]], min_count: int = 2,
                    min_share: float = 0.5) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for latin, indic in pairs:
        lt = clean_tokens(fold(latin))
        it = clean_tokens(fold(indic))
        if len(lt) != len(it):
            continue
        for a, b in zip(lt, it):
            if INDIC_RE.search(b) and not INDIC_RE.search(a):
                counts[b][LEGAL.get(a, a)] += 1
    return _select(counts, min_count, min_share)


def learn_component_map(rows: Iterable[tuple[str, str, str]], min_count: int = 2,
                        min_share: float = 0.5) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for country, latin, indic in rows:
        states = STATE_MAPS.get(country, {})
        lat_states = [k for k in (comp_key(c) for c in latin.split(",")) if k in states]
        ind_comps = [k for k in (comp_key(c) for c in indic.split(",")) if INDIC_RE.search(k)]
        if len(lat_states) == 1 and len(ind_comps) == 1:
            counts[ind_comps[0]][lat_states[0]] += 1
    return _select(counts, min_count, min_share)


def _romanize(tok: str) -> str:
    return _PUNCT.sub("", anyascii(tok).lower()).replace(" ", "")


class Transliterator:
    def __init__(self, token_map: dict[str, str], comp_map: dict[str, str]):
        self.token_map = token_map
        self.comp_map = comp_map

    def token(self, tok: str) -> str:
        hit = self.token_map.get(tok)
        return hit if hit is not None else _romanize(tok)

    def component(self, comp: str) -> str:
        hit = self.comp_map.get(comp)
        if hit is not None:
            return hit
        return " ".join(self.token(t) if INDIC_RE.search(t) else t for t in comp.split())

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"token": self.token_map, "component": self.comp_map},
                                         ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Transliterator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["token"], d["component"])
