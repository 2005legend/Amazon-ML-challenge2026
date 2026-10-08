import numpy as np
import polars as pl

from ber.neural import ADDR_L, CLS, NAME_L, REC, T, UNK, batch_x, encode, g9_context


def _lut():
    lut = np.full(65536, UNK, np.uint8); lut[0] = 0; lut[0x1F] = 3
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 "):
        lut[ord(ch)] = 4 + i
    return lut


def test_encode_pads_each_slot_and_keeps_the_address_tail():
    long_addr = "1 main street " + "x" * 80 + " zip 12345"
    e = encode(pl.DataFrame({"name": ["ab", None], "address": [long_addr, "12 elm st"]}), _lut())
    assert e.shape == (2, REC) and REC == NAME_L + ADDR_L
    assert e[0, :2].tolist() == [4, 5] and (e[0, 2:NAME_L] == 0).all()     # name then padding
    assert (e[1, :NAME_L] == 0).all()                                        # missing name = empty slot
    a = e[0, NAME_L:]
    assert (a != 0).all() and a[ADDR_L - 17] == 3                           # truncated: head, cut marker, last 16 chars
    assert a[-5:].tolist() == [4 + 27 + i for i in (0, 1, 2, 3, 4)]         # "12345" survives


def test_batch_x_lays_out_cls_s1_record_then_target_record():
    arr = np.arange(3 * REC, dtype=np.uint8).reshape(3, REC)
    x = batch_x(arr, np.array([0, 2]), np.array([1, 1]))
    assert x.shape == (2, T) and (x[:, 0] == CLS).all()
    assert (x[1, 1:1 + REC] == arr[2]).all() and (x[1, 1 + REC:] == arr[1]).all()


def test_g9_context_ranks_and_margins_inside_each_s1():
    g = g9_context(pl.DataFrame({"s1_id": ["a", "a", "a", "b"], "tgt_id": ["x", "y", "z", "w"], "g9_logit": [3.0, 1.0, -2.0, 0.5]}))
    r ={t: (rk, gp) for t, rk, gp in g.select("tgt_id", "g9_rank", "g9_gap").rows()}
    assert r["x"] == (1.0, 2.0) and r["y"] == (2.0, -2.0) and r["z"] == (3.0, -5.0) and r["w"] == (1.0, 10.0)


def test_load_g9_averages_the_models_logits(tmp_path):
    from ber.neural import load_g9
    d = tmp_path / "nn"; d.mkdir()
    pl.DataFrame({"s1_id": ["a", "a"], "tgt_id": ["x", "y"], "g9_logit": [2.0, -1.0]}).write_parquet(d / "g9_H.parquet")
    pl.DataFrame({"s1_id": ["a", "a"], "tgt_id": ["y", "x"], "g9_logit": [3.0, 0.0]}).write_parquet(d / "g9_b_H.parquet")
    g = {t: v for t, v in load_g9(tmp_path, "H", ("", "_b")).select("tgt_id", "g9_logit").rows()}
    assert g == {"x": 1.0, "y": 1.0}
    one = load_g9(tmp_path, "H", ("",)).sort("tgt_id")
    assert one["g9_logit"].to_list() == [2.0, -1.0] and one["g9_rank"].to_list() == [1.0, 2.0]
