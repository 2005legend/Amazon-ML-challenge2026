import pytest
from ber.normalize import normalize_name


@pytest.mark.parametrize("raw,core,legal", [
    ("Foot + Ankle  Physicians [PLLC]", "foot ankle physicians", "pllc"),
    ("Keystone-Kensington Inc.", "keystone kensington", "inc"),
    ("5outhern Pioneer Armour LLC", "southern pioneer armour", "llc"),
    ("Jones Cascade-Pa1oma, LLC", "jones cascade paloma", "llc"),
    ("M/s Krishna Technologies Private Límited", "krishna technologies", "ltd pvt"),
    ("Pvt. EFS Print Ventures Ltd.", "efs print ventures", "ltd pvt"),
    ("Trusted Diamond Roman-(L.L.C.)", "trusted diamond roman", "llc"),
    ("HALLY LOCHNER, - 4176446196", "hally lochner", ""),
    ("SMITHMEDICALCENTER.COM", "smithmedicalcenter", ""),
    ("@cardiologyspecialists", "cardiologyspecialists", ""),
    ("THE WOOD TOTAL CIRCLE-LP", "wood total circle", "lp"),
    ("Design Design Global Pirvmate Limited", "design global pirvmate", "ltd"),
    ("ALL PIEDMONT NASDAQ CO CO", "all piedmont nasdaq", "co"),
    ("Établissements Team", "team", "ets"),
    ("ETS 210 HOLDING SAS", "210 holding", "ets sas"),
])
def test_core_and_legal(raw, core, legal):
    out = normalize_name(raw)
    assert out[0] == core
    assert out[1] == core.replace(" ", "")
    assert out[3] == legal


def test_dba_parts():
    core, compact, parts, legal, initials, has_indic = normalize_name("Ariaquo d/b/a Combs Park of Leeds LLC")
    assert parts == "ariaquo|combs park leeds"
    assert core == "ariaquo combs park leeds" and legal == "llc" and initials == "acpl"


def test_empty_name():
    assert normalize_name("") == ("", "", "", "", "", False)


def test_indic_flag_and_translit_hook():
    assert normalize_name("राम मार्केटिंग")[5] is True
    assert normalize_name("राम Traders", translit=lambda t: "ram")[0] == "ram traders"
