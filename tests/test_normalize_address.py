from ber.normalize import normalize_address


def test_us_street_forms_agree():
    a = normalize_address("2312 BATES AVE, PMB 309, SPRINGFIELD, IL", "US")
    b = normalize_address("Springfield, 2312 Bates Avenue, IL", "US")
    c = normalize_address("#2312 Bates Ave, Springfield, Illinois", "US")
    for r in (a, b, c):
        assert r[4] == "il" and r[3] == "2312"
        assert set(r[1].split()) == {"bates", "springfield"}
    assert "309" not in a[2].split()


def test_numbers_ordinals_leading_zeros():
    r = normalize_address("046447 Main Street, 3TH Floor, Centerville, Ohio", "US")
    assert r[3] == "46447" and r[2].split() == ["46447", "3"]
    assert r[0] == "46447 main st 3 fl centerville oh"


def test_number_words_and_saint():
    assert normalize_address("Third Street", "US")[0] == "3 st"
    assert normalize_address("3rd Saint", "US")[0] == "3 st"


def test_null_markers_and_empty():
    r = normalize_address("VA, STTAFFORD, <NULL>, 5 SEVIER SDREET", "US")
    assert "null" not in r[0] and r[4] == "va" and r[3] == "5"
    assert normalize_address("", "India") == ("", "", "", "", "", False, True)
    assert normalize_address("null", "US")[6] is True


def test_india_state_forms_and_house_numbers():
    a = normalize_address("H.NO. ##87, MAHATHUA SANT RAVI DAS NAGAR, AURAI, Uttar Pradesh", "India")
    b = normalize_address("Door No 87, Mahathua Sant Ravi Das Nagar, Auraiya, UP", "India")
    assert a[4] == b[4] == "up" and a[3] == b[3] == "87"


def test_unknown_country_has_no_state():
    r = normalize_address("63 R. DE DIEPPE, LILLE", "Germany")
    assert r[4] == "" and r[3] == "63"
    assert r[0] == "63 rue de dieppe lille"
    assert normalize_address("IL", "Germany")[4] == ""


def test_france_region_and_department_share_a_state():
    a = normalize_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France")
    b = normalize_address("63 Rue de Dieppe, Lille, Nord", "France")
    c = normalize_address("10 R DE GUYENNE, PESSAC, Gironde", "France")
    d = normalize_address("10 Rue de Guyenne, Pessac, Nouvelle-Aquitaine", "France")
    assert a[4] == b[4] == "hdf" and c[4] == d[4] == "naq"
    assert a[0] == "63 rue de dieppe lille hdf"
    assert normalize_address("12 Rue X, Paris", "France")[4] == ""


def test_translit_component_hook():
    r = normalize_address("PUNE, महाराष्ट्र", "India", translit=lambda c: "maharashtra")
    assert r[4] == "mh" and r[5] is True
