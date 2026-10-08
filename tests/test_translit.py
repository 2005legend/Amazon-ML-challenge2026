from ber.normalize import normalize_name
from ber.translit import Transliterator, learn_component_map, learn_token_map


def test_learn_token_map_positional_and_legal_canonical():
    pairs = [("Ram Marketing Private Limited", "राम मार्केटिंग प्राइवेट लिमिटेड"),
             ("Shyam Marketing Pvt Ltd", "श्याम मार्केटिंग प्राइवेट लिमिटेड"),
             ("Ram Traders", "राम ट्रेडर्स")]
    m = learn_token_map(pairs, min_count=2)
    assert m["मार्केटिंग"] == "marketing" and m["राम"] == "ram"
    assert m["प्राइवेट"] == "pvt" and m["लिमिटेड"] == "ltd"
    assert "श्याम" not in m


def test_learn_component_map_uses_state_component():
    rows = [("India", "Pune, Maharashtra", "PUNE, महाराष्ट्र"),
            ("India", "Maharashtra, Mumbai", "महाराष्ट्र, MUMBAI"),
            ("India", "Uttar Pradesh, Agra", "AGRA, उत्तर प्रदेश"),
            ("India", "Agra, Uttar Pradesh", "उत्तर प्रदेश")]
    m = learn_component_map(rows, min_count=2)
    assert m["महाराष्ट्र"] == "maharashtra" and m["उत्तर प्रदेश"] == "uttar pradesh"


def test_fallback_romanizes_unknown_token():
    out = Transliterator({}, {}).token("श्याम")
    assert out and out.isascii() and out.isalnum()


def test_name_with_transliterator_is_ascii():
    t = Transliterator({"प्राइवेट": "pvt", "लिमिटेड": "ltd"}, {})
    core, _, _, legal, _, has_indic = normalize_name("श्याम प्राइवेट लिमिटेड", translit=t.token)
    assert core and core.isascii() and legal == "ltd pvt" and has_indic


def test_component_falls_back_per_token_and_roundtrips(tmp_path):
    t = Transliterator({"राम": "ram"}, {"महाराष्ट्र": "maharashtra"})
    assert t.component("राम नगर").startswith("ram ")
    p = tmp_path / "t.json"
    t.save(p)
    u = Transliterator.load(p)
    assert u.token("राम") == "ram" and u.component("महाराष्ट्र") == "maharashtra"
