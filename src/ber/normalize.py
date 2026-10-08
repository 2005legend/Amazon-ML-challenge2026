"""Pure string normalization for business names and addresses (no I/O)."""
import re
import unicodedata
from typing import Callable, Optional

TokenMap = Optional[Callable[[str], str]]

_SPECIAL = str.maketrans({"œ": "oe", "æ": "ae", "ø": "o", "ß": "ss", "đ": "d", "ł": "l", "ı": "i",
                          "’": "'", "‘": "'", "–": "-", "—": "-", "‌": "", "‍": ""})
INDIC_RE = re.compile(r"[ऀ-ൿ]")
_COMBINING = re.compile(r"[̀-ͯ]")
_PHONE = re.compile(r"\+?\d[\d\s\-]{8,}\d")
_URL = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co|biz|info|io|fr|us)\b")
_MS = re.compile(r"\bm\s*/\s*s\b")
_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|formerly known as|formerly"
                  r"|f/k/a|a/k/a|trading as|t/a)(?=\W|$)")
_PUNCT = re.compile(r"[^0-9a-zऀ-ൿ\s]")
_WS = re.compile(r"\s+")
_OCR = str.maketrans("0134578", "oleastb")

LEGAL = {
    "private": "pvt", "pvt": "pvt", "pvte": "pvt", "limited": "ltd", "ltd": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "plc": "plc", "pc": "pc", "pa": "pa", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "selarl": "selarl", "gmbh": "gmbh",
    "ets": "ets", "etablissements": "ets", "etablissement": "ets",
}
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "d", "l"}


def fold(text: str) -> str:
    """Lowercase, map special letters, strip Latin diacritics (Indic marks are kept)."""
    text = unicodedata.normalize("NFKD", text.translate(_SPECIAL))
    return _COMBINING.sub("", text).lower()


def _ocr_fix(tok: str) -> str:
    if any(c.isdigit() for c in tok) and sum(c.isalpha() for c in tok) >= 3:
        return tok.translate(_OCR)
    return tok


def clean_tokens(text: str, translit: TokenMap = None) -> list[str]:
    """Dots removed, other punctuation to spaces, Indic tokens transliterated, OCR digits fixed."""
    toks = _PUNCT.sub(" ", text.replace(".", "")).split()
    if translit is not None:
        toks = [x for t in toks for x in (translit(t).split() if INDIC_RE.search(t) else [t])]
    return [_ocr_fix(t) for t in toks]


def _name_parts(raw: str) -> list[str]:
    s = _MS.sub(" ", _URL.sub(r" \1 ", _PHONE.sub(" ", fold(raw))))
    return [p for p in _DBA.split(s) if p.strip()] or [""]


def normalize_name(raw: str, translit: TokenMap = None):
    has_indic = bool(INDIC_RE.search(raw))
    legal: set[str] = set()
    cores: list[list[str]] = []
    for part in _name_parts(raw):
        toks = []
        for t in clean_tokens(part, translit):
            if t in LEGAL:
                legal.add(LEGAL[t])
            elif t not in NAME_STOP and (not toks or toks[-1] != t):
                toks.append(t)
        if toks:
            cores.append(toks)
    all_toks = [t for c in cores for t in c]
    core = " ".join(all_toks)
    parts = "|".join(" ".join(c) for c in cores) if len(cores) > 1 else ""
    initials = "".join(t[0] for t in all_toks)
    return core, core.replace(" ", ""), parts, " ".join(sorted(legal)), initials, has_indic


# ---------------------------------------------------------------- addresses
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la",
    "puducherry": "py", "pondicherry": "py", "chandigarh": "ch", "andaman and nicobar islands": "an",
    "dadra and nagar haveli and daman and diu": "dn", "lakshadweep": "ld",
    "ts": "tg", "or": "od", "ct": "cg", "ut": "uk",
}


def _with_codes(m: dict[str, str]) -> dict[str, str]:
    return {**m, **{v: v for v in m.values()}}


# Metropolitan French regions and their departements (general geography, like the US state codes).
# Keys are in normalized component form (accents stripped, hyphens/apostrophes -> spaces). The
# departement "Paris" is left out on purpose: as a component it almost always names the city.
_FR_REGIONS = {
    "ara": ("auvergne rhone alpes", ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire",
                                     "haute loire", "puy de dome", "rhone", "savoie", "haute savoie"]),
    "bfc": ("bourgogne franche comte", ["cote d or", "doubs", "jura", "nievre", "haute saone", "saone et loire",
                                        "yonne", "territoire de belfort"]),
    "bre": ("bretagne", ["cotes d armor", "finistere", "ille et vilaine", "morbihan"]),
    "cvl": ("centre val de loire", ["cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"]),
    "cor": ("corse", ["corse du sud", "haute corse"]),
    "ges": ("grand est", ["ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse", "moselle",
                          "bas rhin", "haut rhin", "vosges"]),
    "hdf": ("hauts de france", ["aisne", "nord", "oise", "pas de calais", "somme"]),
    "idf": ("ile de france", ["seine et marne", "yvelines", "essonne", "hauts de seine", "seine st denis",
                              "seine saint denis", "val de marne", "val d oise"]),
    "nor": ("normandie", ["calvados", "eure", "manche", "orne", "seine maritime"]),
    "naq": ("nouvelle aquitaine", ["charente", "charente maritime", "correze", "creuse", "dordogne", "gironde",
                                   "landes", "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne",
                                   "haute vienne"]),
    "occ": ("occitanie", ["ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot", "lozere",
                          "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"]),
    "pdl": ("pays de la loire", ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"]),
    "pac": ("provence alpes cote d azur", ["alpes de haute provence", "hautes alpes", "alpes maritimes",
                                           "bouches du rhone", "var", "vaucluse"]),
}
FR_STATES = {name: code for code, (region, depts) in _FR_REGIONS.items() for name in [region, *depts]}

STATE_MAPS: dict[str, dict[str, str]] = {"US": _with_codes(US_STATES), "India": _with_codes(IN_STATES),
                                         "France": FR_STATES}

ADDR_MAP = {
    "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "avenida": "ave", "road": "rd",
    "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct", "circle": "cir", "boulevard": "blvd", "bd": "blvd",
    "bld": "blvd", "boul": "blvd", "place": "pl", "terrace": "ter", "parkway": "pkwy", "pky": "pkwy",
    "highway": "hwy", "square": "sq", "trail": "trl", "crossing": "xing", "point": "pt", "mount": "mt",
    "mountain": "mtn", "fort": "ft", "heights": "hts", "junction": "jct", "expressway": "expy",
    "freeway": "fwy", "center": "ctr", "centre": "ctr", "alley": "aly", "plaza": "plz", "route": "rte",
    "chemin": "ch", "impasse": "imp", "allee": "all", "faubourg": "fbg", "cours": "crs", "r": "rue",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "apartment": "apt", "suite": "ste", "building": "bldg",
    "floor": "fl", "flr": "fl", "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
}
ADDR_DROP = {"h", "hn", "hno", "house", "door", "dno", "no", "number", "num", "nos"}
ADDR_GENERIC = set(ADDR_MAP.values()) | {"unit", "apt", "ste", "fl", "bldg", "flat", "nr", "near", "opp"}
_NULLS = re.compile(r"<\s*null\s*>|\bnull\b|\bnone\b|\bn/a\b")
_POBOX = re.compile(r"\bp\s*o\s*box\s*\d+|\bpmb\s*\d+")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_DIGITS = re.compile(r"\d+")
_COMP_SPLIT = re.compile(r"[,;|]")


def _addr_token(t: str) -> str:
    return str(int(t)) if t.isdigit() else ADDR_MAP.get(t, t)


def normalize_address(raw: str, country: str, translit: TokenMap = None):
    has_indic = bool(INDIC_RE.search(raw))
    states = STATE_MAPS.get(country, {})
    text = _POBOX.sub(" ", _NULLS.sub(" ", fold(raw)))
    comps, state = [], ""
    for comp in _COMP_SPLIT.split(text):
        comp = _ORDINAL.sub(r"\1", comp.replace(".", ""))
        comp = _WS.sub(" ", _PUNCT.sub(" ", comp)).strip()
        if not comp:
            continue
        if translit is not None and INDIC_RE.search(comp):
            comp = translit(comp)
        code = states.get(comp)
        if code:
            state = code
            comps.append(code)
            continue
        toks = [_addr_token(t) for t in comp.split() if t not in ADDR_DROP]
        if toks:
            comps.append(" ".join(toks))
    norm = " ".join(comps)
    nums = [str(int(m)) for m in _DIGITS.findall(norm)]
    words = [t for t in norm.split() if not t.isdigit() and t not in ADDR_GENERIC and t != state]
    return (norm, " ".join(dict.fromkeys(words)), " ".join(dict.fromkeys(nums)),
            nums[0] if nums else "", state, has_indic, not norm)
