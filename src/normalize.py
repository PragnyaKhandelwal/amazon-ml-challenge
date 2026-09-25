"""Deterministic text normalisation of business names and addresses.

No external lookups: only generic abbreviation / legal-form / ordinal tables.
Every source goes through the same functions, so an unknown variant simply stays
as-is on both sides.  Runs over *unique* strings with a process pool.
"""
import re
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

# --------------------------------------------------------------------------- names
LEGAL_CANON = {
    "private": "private", "pvt": "private", "pte": "private", "prvt": "private", "pvtltd": "private limited",
    "limited": "limited", "ltd": "limited", "ltda": "limited", "ld": "limited",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "coy": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "pa": "pa",
    "sa": "sa", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "snc": "snc",
    "sci": "sci", "selarl": "selarl", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "opc": "opc", "public": "public",
}
LEGAL_SET = set(LEGAL_CANON.values()) | {"private limited"}
# dotted / spaced legal forms joined before tokenisation
_LEGAL_JOIN = [(re.compile(p), r) for p, r in [
    (r"\bl\s*\.?\s*l\s*\.?\s*c\b\.?", " llc "), (r"\bl\s*\.?\s*l\s*\.?\s*p\b\.?", " llp "),
    (r"\bp\s*\.?\s*l\s*\.?\s*l\s*\.?\s*c\b\.?", " pllc "), (r"\bp\s*\.\s*c\b\.?", " pc "),
    (r"\bp\s*\.\s*a\b\.?", " pa "), (r"\bl\s*\.\s*p\b\.?", " lp "),
    (r"\bs\s*\.\s*a\s*\.\s*r\s*\.\s*l\b\.?", " sarl "), (r"\bs\s*\.\s*a\s*\.\s*s\b\.?", " sas "),
    (r"\be\s*\.\s*u\s*\.\s*r\s*\.\s*l\b\.?", " eurl "), (r"\bs\s*\.\s*a\b\.?", " sa "),
    (r"\bs\s*\.\s*n\s*\.\s*c\b\.?", " snc "), (r"\bs\s*\.\s*c\s*\.\s*i\b\.?", " sci "),
    (r"\bpvt\s*\.?\s*ltd\b", " private limited "), (r"\bpvt\.", " pvt "),
]]
NAME_PREFIX_JUNK = {"the", "dr", "sri", "shri", "shree", "sree", "mr", "mrs", "ms", "smt", "m s",
                    "messrs", "a", "an", "le", "la", "les", "l"}
NAME_ABBR = {
    "intl": "international", "natl": "national", "mfg": "manufacturing", "svc": "services",
    "svcs": "services", "tech": "technologies", "technology": "technologies",
    "assoc": "associates", "bros": "brothers", "engg": "engineering", "grp": "group",
    "hosp": "hospital", "inds": "industries", "ind": "industries", "industry": "industries",
    "mgmt": "management", "mktg": "marketing", "pharma": "pharmaceuticals",
    "sys": "systems", "univ": "university", "ent": "enterprises", "ents": "enterprises",
    "enterprise": "enterprises", "hldgs": "holdings", "ctr": "center", "centre": "center",
    "cntr": "center", "n": "and", "et": "and", "ets": "etablissements", "ste": "societe",
    "st": "saint", "mt": "mount", "service": "services", "solution": "solutions",
    "consultant": "consultants", "holding": "holdings", "partner": "partners",
    "product": "products", "system": "systems", "brother": "brothers", "export": "exports",
    "venture": "ventures", "infratech": "infratech",
}
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "l", "d", "a", "an", "et"}
_DBA = re.compile(r"\b(?:trading as|doing business as|formerly known as|formerly|t\s*/\s*a|"
                  r"d\s*/\s*b\s*/\s*a|dba|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka)\b\.?")
_WEB = re.compile(r"\|\s*(?:https?://)?(?:www\.)?([a-z0-9\-]+)(?:\.[a-z]{2,4}){1,2}\S*")
_DOMAIN = re.compile(r"^#?\s*(?:www\.)?([a-z0-9\-]+)((?:\.[a-z]{2,4}){1,2})?$")
_NONAN = re.compile(r"[^a-z0-9]+")
_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "8": "b", "7": "t", "|": "l"})
_L_AS_I = re.compile(r"^l(?=[bcdfgkmnpqrstvwxz])")


def ascii_lower(s):
    if not s:
        return ""
    if not s.isascii():
        s = unidecode(s)
    return s.lower()


def fix_token(t):
    """OCR-style repairs inside alphabetic tokens (capita1 -> capital, lnc -> inc)."""
    if not t.isalnum() or t.isdigit() or t.isalpha():
        return _L_AS_I.sub("i", t) if t.isalpha() and len(t) > 2 else t
    n_d = sum(c.isdigit() for c in t)
    if n_d <= 2 and len(t) - n_d >= 2 and not re.fullmatch(r"\d+(st|nd|rd|th|er|e|eme|b|a|bis)", t):
        return t.translate(_OCR)
    return t


def _name_tokens(s):
    for p, r in _LEGAL_JOIN:
        s = p.sub(r, s)
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "")
    toks = []
    for t in _NONAN.split(s):
        if not t:
            continue
        t = fix_token(t)
        t = NAME_ABBR.get(t, t)
        t = LEGAL_CANON.get(t, t)
        toks.extend(t.split())
    # consecutive duplicate tokens ("software software")
    out = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    while len(out) > 1 and out[0] in NAME_PREFIX_JUNK:
        out = out[1:]
    return out


def _core(toks):
    c = [t for t in toks if t not in LEGAL_SET and t not in NAME_STOP]
    return c or [t for t in toks if t not in NAME_STOP] or toks


def legal_category(toks):
    s = set(toks)
    if "private" in s and "limited" in s:
        return "pvtltd"
    for k in ("llp", "llc", "pllc", "plc", "lp", "pc", "pa", "sarl", "sas", "sasu", "eurl",
              "snc", "sci", "sa", "gmbh"):
        if k in s:
            return k
    if "inc" in s or "corp" in s:
        return "inc"
    if "limited" in s:
        return "ltd"
    if "co" in s:
        return "co"
    if "private" in s:
        return "pvt"
    return ""


def normalize_name(raw):
    """-> (nm, core, alt, legal, is_domain)"""
    s = ascii_lower(raw)
    s = re.sub(r"^\s*m\s*/\s*s\b\.?", " ", s)
    alt = []
    m = _WEB.search(s)
    if m:
        alt.append(m.group(1).replace("-", ""))
        s = s[:m.start()] + " " + s[m.end():]
    is_dom = 0
    st = s.strip().strip(".,;:*-<>!~ ").strip()
    dm = _DOMAIN.match(st)
    if dm and (dm.group(2) or st.startswith("#")) and " " not in st:
        is_dom = 1
        s = dm.group(1).replace("-", "")
    parts = [p for p in _DBA.split(s) if p and p.strip()]
    main = parts[0] if parts else s
    if len(parts) > 1:
        # synthetic brand usually comes first, real name after the connector: keep the longer
        # side as the primary and the other as an alternative
        cand = [(_name_tokens(p), p) for p in parts]
        cand.sort(key=lambda x: -len(_core(x[0])))
        main = cand[0][1]
        for tk, _ in cand[1:]:
            alt.append(" ".join(_core(tk)))
    toks = _name_tokens(main)
    core = _core(toks)
    return (" ".join(toks), " ".join(core), " | ".join(a for a in alt if a),
            legal_category(toks), is_dom)


# ------------------------------------------------------------------------ addresses
ADDR_CANON = {
    "street": "st", "str": "st", "st": "st", "saint": "st", "ste": "ste", "sreet": "st", "stree": "st",
    "road": "rd", "rd": "rd", "raod": "rd", "avenue": "av", "ave": "av", "av": "av", "avn": "av",
    "avenu": "av", "boulevard": "bd", "blvd": "bd", "bd": "bd", "boul": "bd", "bld": "bd",
    "lane": "ln", "ln": "ln", "drive": "dr", "dr": "dr", "drv": "dr", "court": "ct", "ct": "ct",
    "crt": "ct", "place": "pl", "pl": "pl", "plaza": "plz", "plz": "plz", "square": "sq",
    "sq": "sq", "terrace": "ter", "ter": "ter", "terr": "ter", "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "trail": "trl", "trl": "trl",
    "route": "rte", "rte": "rte", "rt": "rte", "way": "way", "wy": "way", "suite": "ste",
    "apartment": "apt", "apt": "apt", "appt": "apt", "apts": "apt", "unit": "unit",
    "floor": "fl", "fl": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg", "room": "rm",
    "rm": "rm", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt",
    "heights": "hts", "hts": "hts", "junction": "jct", "jct": "jct", "north": "n", "n": "n",
    "south": "s", "s": "s", "east": "e", "e": "e", "west": "w", "w": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "near": "nr", "nr": "nr",
    "opposite": "opp", "opp": "opp", "oppo": "opp", "behind": "bhd", "bhd": "bhd",
    "station": "stn", "stn": "stn", "sector": "sec", "sec": "sec", "sect": "sec", "phase": "ph",
    "ph": "ph", "block": "blk", "blk": "blk", "complex": "cmplx", "cmplx": "cmplx",
    "colony": "col", "col": "col", "nagar": "ngr", "ngr": "ngr", "extension": "extn",
    "extn": "extn", "ext": "extn", "society": "soc", "soc": "soc", "district": "dist",
    "dist": "dist", "distt": "dist", "village": "vill", "vill": "vill", "industrial": "indl",
    "indl": "indl", "estate": "est", "est": "est", "market": "mkt", "mkt": "mkt", "main": "main",
    "cross": "crs", "crs": "crs", "ground": "gr", "gf": "gr", "marg": "marg", "mg": "marg",
    "rue": "rue", "r": "rue", "allee": "all", "allees": "all", "all": "all", "impasse": "imp",
    "imp": "imp", "chemin": "ch", "che": "ch", "ch": "ch", "chem": "ch", "faubourg": "fbg",
    "fbg": "fbg", "quai": "qu", "cours": "crs", "chaussee": "chs", "route_": "rte",
    "city": "", "cdp": "", "township": "", "tonship": "", "twp": "", "of": "", "the": "",
    "no": "", "number": "", "num": "", "nbr": "", "null": "", "none": "", "nan": "",
    "house": "", "hno": "", "h": "", "door": "", "plot": "", "plno": "", "flno": "", "shop": "",
    "po": "", "box": "", "pin": "", "pincode": "", "code": "", "de": "", "du": "", "des": "",
    "la": "", "le": "", "les": "", "d": "", "l": "", "c": "", "o": "", "and": "",
    "county": "cnty", "cnty": "cnty", "ndeg": "", "bis": "", "ter_": "",
}
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "premier": "1",
    "premiere": "1", "deuxieme": "2", "troisieme": "3",
}
REGION_MULTI = {  # multi-word regions -> single canonical token (generic abbreviation table)
    "new york": "ny", "new jersey": "nj", "new mexico": "nm", "new hampshire": "nh",
    "north carolina": "nc", "south carolina": "sc", "north dakota": "nd", "south dakota": "sd",
    "west virginia": "wv", "rhode island": "ri", "district of columbia": "dc",
    "tamil nadu": "tn", "uttar pradesh": "up", "madhya pradesh": "mp", "andhra pradesh": "ap",
    "himachal pradesh": "hp", "arunachal pradesh": "arp", "west bengal": "wb",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "new delhi": "newdelhi",
}
REGION_SINGLE = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co_", "connecticut": "ct_", "delaware": "de_", "florida": "fl_", "georgia": "ga",
    "hawaii": "hi", "idaho": "id_", "illinois": "il", "indiana": "in_", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la_", "maine": "me_", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt_", "nebraska": "ne_", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or_", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy", "maharashtra": "mh", "karnataka": "ka", "gujarat": "gj", "rajasthan": "rj",
    "kerala": "kl", "haryana": "hr", "punjab": "pb", "bihar": "br", "odisha": "od",
    "orissa": "od", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "telangana": "ts", "tg": "ts", "assam": "as_", "goa": "goa", "tripura": "tr_",
    "meghalaya": "ml_", "manipur": "mn_", "nagaland": "nl_", "mizoram": "mz", "sikkim": "sk",
    "chandigarh": "chd", "puducherry": "py", "pondicherry": "py", "bombay": "mumbai",
    "bangalore": "bengaluru", "bengalore": "bengaluru", "banglore": "bengaluru",
    "madras": "chennai", "calcutta": "kolkata", "gurgaon": "gurugram", "poona": "pune",
    "baroda": "vadodara", "trivandrum": "thiruvananthapuram", "cochin": "kochi",
    "mysore": "mysuru", "nyc": "ny", "delhi": "delhi", "dl": "delhi",
}
# 2-letter codes that collide with ordinary words only in the canonical output space
_ADDR_SPLIT = re.compile(r"[^a-z0-9]+")
_ORD_NUM = re.compile(r"^(\d+)(?:st|nd|rd|th|er|eme|e|re)$")
_MULTI_RE = re.compile(r"\b(" + "|".join(sorted(REGION_MULTI, key=len, reverse=True)) + r")\b")
_POSTAL = re.compile(r"\b(\d{3})\s(\d{3})\b")
_ZIP4 = re.compile(r"\b(\d{5})-\d{4}\b")


def normalize_address(raw):
    """-> (ad, words, nums, postal, hn)"""
    s = ascii_lower(raw)
    if not s:
        return ("", "", "", "", "")
    s = s.replace("<null>", " ").replace("n/a", " ")
    s = _ZIP4.sub(r"\1", s)
    s = _POSTAL.sub(r"\1\2", s)
    s = s.replace("c/o", " co ").replace("b/h", " bhd ").replace("s/o", " so ").replace("w/o", " wo ")
    s = _MULTI_RE.sub(lambda m: " " + REGION_MULTI[m.group(1)] + " ", s)
    toks, nums, hn = [], [], ""
    for comp in s.split(","):
        ctoks = []
        for t in _ADDR_SPLIT.split(comp):
            if not t:
                continue
            m = _ORD_NUM.match(t)
            if m:
                t = m.group(1)
            elif not t.isdigit() and any(c.isdigit() for c in t):
                # "12b", "a48", "1-98" pieces: split letters / digits
                sub = re.findall(r"\d+|[a-z]+", t)
                for x in sub:
                    x = (x.lstrip("0") or "0") if x.isdigit() else ADDR_CANON.get(x, x)
                    if x:
                        ctoks.append(x)
                continue
            t = ORDINAL_WORDS.get(t, t)
            if t.isdigit():
                t = t.lstrip("0") or "0"
            else:
                t = ADDR_CANON.get(t, t)
                t = REGION_SINGLE.get(t, t)
            if t:
                ctoks.append(t)
        if not hn:
            d = [x for x in ctoks if x.isdigit()]
            if d and len(d[0]) <= 5:
                hn = d[0]
        toks.extend(ctoks)
    postal = ""
    for t in toks:
        if t.isdigit():
            if len(t) == 6:
                postal = t
            else:
                nums.append(t)
    words = [t for t in toks if not t.isdigit()]
    nums = sorted(set(nums), key=lambda x: (len(x), x))
    return (" ".join(toks), " ".join(words), " ".join(nums), postal, hn)


# ------------------------------------------------------------------------ phonetic
_TL_RULES = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"),
             ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("w", "v"), ("z", "j"),
             ("q", "k"), ("x", "ks"), ("y", "i"), ("c", "k")]
_DOUBLE = re.compile(r"(.)\1+")
_VOW = re.compile(r"[aeiou]")


def skeleton(t):
    """Consonant skeleton robust to transliteration / vowel typos (krishna ~ kRssnn)."""
    if not t or t.isdigit():
        return t
    for a, b in _TL_RULES:
        t = t.replace(a, b)
    t = _DOUBLE.sub(r"\1", t)
    return t[0] + _VOW.sub("", t[1:])


# ---------------------------------------------------------------------- frame api
def _norm_name_chunk(xs):
    return [normalize_name(x) for x in xs]


def _norm_addr_chunk(xs):
    return [normalize_address(x) for x in xs]


def _pmap(fn, values, pool):
    if pool is None or len(values) < 50_000:
        return fn(values)
    n = max(1, len(values) // 32)
    chunks = [values[i:i + n] for i in range(0, len(values), n)]
    res = pool.map(fn, chunks, chunksize=1)
    return [r for c in res for r in c]


def make_pool(workers):
    return Pool(workers) if workers > 1 else None


def normalize_frame(df, translit=None, pool=None):
    """df: entity_id, business_name, business_address, country, src -> normalised frame."""
    if translit is not None:
        df = df.with_columns(name_rom=translit.convert_series(pl.col("business_name")),
                             addr_rom=translit.convert_series(pl.col("business_address")))
    else:
        df = df.with_columns(name_rom=pl.col("business_name"), addr_rom=pl.col("business_address"))
    un = df["name_rom"].unique().to_list()
    res = _pmap(_norm_name_chunk, un, pool)
    nmap = pl.DataFrame({"name_rom": un, "nm": [r[0] for r in res], "core": [r[1] for r in res],
                         "alt": [r[2] for r in res], "legal": [r[3] for r in res],
                         "is_dom": pl.Series([r[4] for r in res], dtype=pl.Int8)})
    ua = df["addr_rom"].unique().to_list()
    res = _pmap(_norm_addr_chunk, ua, pool)
    amap = pl.DataFrame({"addr_rom": ua, "ad": [r[0] for r in res], "adw": [r[1] for r in res],
                         "nums": [r[2] for r in res], "postal": [r[3] for r in res],
                         "hn": [r[4] for r in res]})
    df = df.join(nmap, on="name_rom", how="left").join(amap, on="addr_rom", how="left")
    # phonetic skeleton of the core name (token-wise)
    ucore = df["core"].unique().to_list()
    sk = pl.DataFrame({"core": ucore, "sk": [" ".join(skeleton(t) for t in c.split()) for c in ucore]})
    df = df.join(sk, on="core", how="left")
    df = df.with_columns(
        country_n=pl.col("country").str.to_lowercase().str.strip_chars(),
        nsp=pl.col("core").str.replace_all(" ", ""),
        ntok=pl.col("core").str.count_matches(r"\S+").cast(pl.Int16),
        first=pl.col("core").str.extract(r"^(\S+)", 1).fill_null(""),
        acro=pl.col("core").str.extract_all(r"\b[a-z]").list.join("")
            .map_elements(lambda x: x if len(x) >= 2 else "", return_dtype=pl.String),
        is_indic=pl.col("business_name").str.contains(r"[ऀ-෿]").cast(pl.Int8),
    )
    return df.drop("name_rom", "addr_rom")
