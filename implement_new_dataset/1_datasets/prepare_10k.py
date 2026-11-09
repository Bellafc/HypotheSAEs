"""Prepare the 10-K Corpus MD&A sections for HypotheSAEs (filing years 2010-2013), two stages.

Data: 10-K Corpus (Kogan et al. 2009), 1996-2013 extension by Tsai et al. (2016): data/10k/raw/mda/<year>.mda.tgz
  and data/10k/raw/ex/<year>.{meta.txt, logvol.-12.txt, logvol.+12.txt}.

Stage "prepare" (step A): process ALL 2010-2013 MD&A reports.
  1. Clean the MD&A to prose paragraphs (drop tables, page numbers, table-of-contents fragments); drop reports with
     < 250 characters of prose.
  2. Look-ahead-bias masking, two passes.
     Rule pass (regex, before chunking), 7 kinds of spans:
       own company name                                   -> [COMPANY]
       web addresses                                      -> [URL]
       ticker symbols                                     -> [TICKER]
       other companies (Capitalized Name + Inc/Corp/...)  -> [OTHER_COMPANY]
       titled person names (Mr./Ms./Mrs./Dr.)             -> [PERSON]
       full and numeric dates                             -> [DATE]
       years, relative to the report's fiscal year T      -> [YEAR_T], [YEAR_T-1], [YEAR_T+1], ...
     NER pass (spaCy en_core_web_sm, after chunking), 2 entity types:
       ORG -> [ORG], PERSON -> [PERSON]   (whitelisted names such as the FASB / SEC are kept)
  3. Split into sentences (pysbd), chunk every 10 sentences, keep at most the first 5 chunks per report, drop chunks
     < 250 characters.
  Output: data/10k/reports.json (one row per report, list of chunks; rule-only masking in chunks_rulemask, no masking
  in chunks_unmasked), prep_info.json.

Stage "split" (step B): all 2010-2013 reports of a firm go into one set, so every set covers
- Firms (CIK) never cross sets (all years of a firm in one set).
  holdout: ~2,000 chunks, one random chunk from each of ~2,000 reports
  val    : ~2,000 chunks, all chunks of its firms' reports
  train  : all chunks of the remaining firms
  
- label y = (logvol - train mean of logvol in that filing year) / train SD of that   (raw log post-filing volatility,
  year fixed effects estimated on train). prevol is kept as prevol_c (same scale), a covariate.
  Output: data/10k/{train,val,holdout}.json, split_info.json
  ("n_hold_firms" in split_info.json is the TARGET number of holdout reports; the firm count is n_hold_firms_actual.)

Example row (data/10k/train.json, row 4181):
    {"key": "127150308-10-K-20130312", "year": 2013, "T": 2012, "cik": "350199", "chunk_idx": 0,
     "text": "We are a nationwide, mall- and web-based specialty retailer of lifestyle sportswear and dresses targeting
              style-conscious women. ... all of which are sold under our [COMPANY] brand, although during fiscal
              [YEAR_T+1] we intend to explore the possibility of also offering products under labels other than
              [COMPANY]. ... As of [DATE], we operated 260 stores ...",  (210 words)
     "prevol": -3.15383, "logvol": -3.40816, "y": 0.960, "prevol_c": 1.385}
  text = chunk 0 (10 sentences) of the MD&A of a women's apparel retailer's 10-K filed in March 2013 (fiscal year
         T = 2012); company/brand name, date and year masked (checked against chunks_unmasked: no wrong masks)
  y    = target to predict (regression): (logvol - train mean of logvol in 2013) / train SD
       = (-3.408 - (-3.857)) / 0.467 = 0.960, i.e. 0.96 SD more volatile than the average 2013 filer.
         logvol / prevol = log volatility of daily returns in the 12 months after / before filing (from the corpus);
         prevol_c = prevol on the same scale (used by the "history" baseline of kogan_baseline_10k.py).

Usage:
  python implement_new_dataset/1_datasets/prepare_10k.py                          # both stages
  python implement_new_dataset/1_datasets/prepare_10k.py --stage split [--n_hold 2000 --n_val 2000 --seed 42]
"""
import argparse
import collections
import json
import os
import re
import tarfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pysbd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
RAW = os.path.join(ROOT, "data/10k/raw")
OUT = os.path.join(ROOT, "data/10k")
YEARS = [2010, 2011, 2012, 2013]
PREFIX_CHARS = 25000  # only the beginning of MD&A is needed for 5 chunks x 10 sentences
COLS = ["key", "year", "T", "cik", "prevol", "logvol"]

# ------------------------------------------------------------------ cleaning
def clean_mda(raw):
    """keep prose paragraphs: blocks separated by blank lines, joined, filtered by simple text statistics."""
    raw = raw.replace("\r", "")
    keep = []
    for block in re.split(r"\n\s*\n", raw):
        b = re.sub(r"\s+", " ", block).strip()
        b = re.sub(r"\[[^\]]*\.(jpg|gif|png)\]", " ", b, flags=re.I).strip()
        if len(b) < 80 or len(b.split()) < 12:
            continue
        letters = sum(c.isalpha() for c in b); digits = sum(c.isdigit() for c in b)
        nonspace = max(sum(not c.isspace() for c in b), 1)
        if letters / nonspace < 0.6 or digits / nonspace > 0.15:
            continue  # tables / numeric lines
        if not re.search(r"[.!?][\"”’)]?$|[.!?] ", b):
            continue  # headings / TOC fragments without sentences
        if re.match(r"^(item\s+7|management.?s discussion and analysis)", b, re.I) and len(b) < 250:
            continue
        keep.append(b)
    return "\n".join(keep)


# ------------------------------------------------------------------ masking (rule pass)
# legal suffixes that may follow a company name; also matches state tags like /DE/
SUFFIXES = r"(?:incorporated|inc|corporation|corp|company|co|limited|ltd|llc|l\.?l\.?c|l\.?p|lp|plc|holdings?|group|trust|bancorp(?:oration)?|/\w+/?)"
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"


def name_variants(name):
    """Build the set of strings that count as the report's OWN company name (masked as [COMPANY] in mask_text).

    Input: the company name registered with the SEC (EDGAR), taken from the meta file (load_meta), e.g.
    'ACME WIDGETS HOLDINGS INC /DE/'. It is NOT detected in the text; the filer's own name is known per report.

    Example: 'ACME WIDGETS HOLDINGS INC /DE/'
        -> {'ACME WIDGETS HOLDINGS INC', 'ACME WIDGETS HOLDINGS', 'ACME WIDGETS', 'ACME'}
    """
    # step 1: clean the registered name
    n = re.sub(r"/[A-Za-z]{2,4}/?", " ", name)                       # drop state tags: INC/VA/, CORP /OH/
    n = re.sub(r"(?<=[A-Za-z]{3})(INC|CORP|LTD)\.?$", r" \1", n)      # split a glued suffix: ENTERTAINMENTINC
    n = re.sub(r"\s+", " ", n).strip(" ,.")                           # collapse spaces, strip edge punctuation
    out = {n}
    # step 2: strip one trailing legal suffix (and a leading 'the') per round, up to 4 rounds;
    # every intermediate form is a variant ('... HOLDINGS INC' -> '... HOLDINGS' -> 'ACME WIDGETS')
    cur = n
    for _ in range(4):
        new = re.sub(rf"[\s,]+{SUFFIXES}\.?$", "", cur, flags=re.I).strip(" ,.&")
        new = re.sub(r"^the\s+", "", new, flags=re.I)
        if new == cur or not new:
            break
        out.add(new); cur = new
    # step 3: shorter forms companies use for themselves, built from the suffix-free name:
    #   - first two words if the name has >= 3 words ("Streamline Health" for "Streamline Health Solutions")
    #   - first word alone if the name has >= 2 words and that word has >= 4 letters ("Streamline");
    #     single words that are common English words are skipped later, in mask_text
    words = cur.split()
    if len(words) >= 3:
        out.add(" ".join(words[:2]))
    if len(words) >= 2 and len(words[0]) >= 4:
        out.add(words[0])
    # step 4: drop variants shorter than 3 characters
    return {v for v in out if len(v) >= 3}


def year_token(y, T):
    """year y -> token relative to fiscal year T: T -> [YEAR_T], T-1 -> [YEAR_T-1], T+1 -> [YEAR_T+1]."""
    d = y - T
    return "[YEAR_T]" if d == 0 else f"[YEAR_T{d:+d}]"


def detect_fiscal_year(text, filing_year, filing_month):
    """fiscal year T = whichever of filing_year-1 / filing_year is mentioned more; ties -> by filing month."""
    c = collections.Counter(int(x) for x in re.findall(r"\b(20\d\d|19\d\d)\b", text))
    a, b = c.get(filing_year - 1, 0), c.get(filing_year, 0)
    if a != b:
        return filing_year - 1 if a > b else filing_year
    return filing_year if filing_month >= 7 else filing_year - 1


def mask_text(text, own_variants, T, common_words):
    """Rule pass: replace 7 kinds of spans with mask tokens. Order matters (company names before the generic
    'other company' rule, dates before bare years). Returns the masked text and replacement counts per kind.

    own_variants : output of name_variants() for this report's company
    T            : fiscal year of the report (detect_fiscal_year)
    common_words : lower-case words that appear in > 1% of all MD&As (computed in prepare)
    """
    counts = collections.Counter()

    def sub(pattern, repl, s, key, flags=0):
        s2, n = re.subn(pattern, repl, s, flags=flags)
        counts[key] += n
        return s2

    s = text

    # (1) own company name -> [COMPANY]
    #   - variants are tried longest first, so 'Acme Widgets' is replaced before 'Acme' could break it up
    #   - a one-word variant that is a common word (in > 1% of MD&As) is skipped, e.g. 'American'
    #   - each variant becomes a case-insensitive regex that also accepts:
    #       * spaces / commas / periods / hyphens between its words ('Acme-Widgets', 'Acme, Widgets')
    #       * a possessive "'s" ('Acme's')
    #       * any trailing legal suffix ('Acme Widgets Corporation', even if the registered name says INC)
    #       * a trailing period only when the sentence continues ('Acme Inc., which ...'), not at sentence end
    #   - (?<![\w]) / (?![\w]) require word boundaries, so a variant never matches inside another word
    for v in sorted(own_variants, key=len, reverse=True):
        if " " not in v and v.lower() in common_words:
            continue
        pat = r"(?<![\w])" + r"[\s,.\-]+".join(re.escape(w) for w in v.split()) + r"(?:['’]?s(?=\s))?(?:[\s,]+" + SUFFIXES + r")?(?:\.(?=(?-i:\s*[a-z,;'])))?(?![\w])"
        s = sub(pat, "[COMPANY]", s, "own_company", re.I)

    # (2) web addresses -> [URL]   (they often contain the company name)
    s = sub(r"\b(?:https?://)?(?:www\.)?[\w\-]+(?:\.[\w\-]+)*\.(?:com|net|org|biz|us|info)(?:/\S*)?", "[URL]", s, "url",
            re.I)

    # (3) ticker symbols -> [TICKER], only in explicit contexts (1-5 capital letters would otherwise hit acronyms):
    #   'NYSE: ACME' / 'Nasdaq:ACME', '(trading|ticker|stock) symbol "ACME"', 'under the symbol ACME'
    s = sub(r"(\b(?:NYSE|NASDAQ|Nasdaq|AMEX|Amex|OTCBB|OTC)\s?:\s?)[A-Z]{1,5}(?:\.[A-Z])?\b", r"\1[TICKER]", s, "ticker")
    s = sub(r"((?:trading|ticker|stock)\s+symbol\s*(?:of\s+)?[\"“']?)[A-Z]{1,5}(?:\.[A-Z])?\b", r"\1[TICKER]", s, "ticker", re.I)
    s = sub(r"(\bunder\s+the\s+symbol\s+[\"“']?)[A-Z]{1,5}(?:\.[A-Z])?\b", r"\1[TICKER]", s, "ticker")

    # (4) other companies -> [OTHER_COMPANY]: 1-6 Capitalized words (may contain 'of' / 'and' / '&')
    #   followed by a legal suffix, e.g. 'Wells Fargo Bank, N.A.', 'Johnson & Johnson Co.'
    #   NW: words that never start a name, so 'The Company' / 'Our Operating Company' are not matched
    NW = r"(?!(?:The|Our|This|That|These|Such|Each|Parent|Operating|Accounting|For|In|As|At|On|Of)\b)"
    s = sub(rf"\b{NW}[A-Z][\w&.'\-]*(?:\s+(?:of\s+|and\s+|&\s+)?{NW}[A-Z][\w&.'\-]*){{0,5}},?\s+"
            r"(?:Inc|Corp|Corporation|Company|Co|Ltd|LLC|L\.P|LP|plc|N\.A|S\.A|AG|N\.V|GmbH)\b\.?",
            "[OTHER_COMPANY]", s, "other_company")

    # (5) titled person names -> [PERSON]: Mr./Ms./Mrs./Dr. + 1-2 Capitalized words ('Mr. John Smith')
    #   (untitled names are left to the NER pass)
    s = sub(r"\b(?:Mr|Ms|Mrs|Dr)\.?\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?", "[PERSON]", s, "person")

    # (6) dates -> [DATE]: 'March 12, 2013' / 'Mar. 12th 2013', and numeric '3/12/2013' / '3/12/13'
    #   (done before (7) so the year inside a full date is not turned into a [YEAR_T...] token)
    s = sub(rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d\d\b", "[DATE]", s, "date")
    s = sub(r"\b\d{1,2}/\d{1,2}/(?:19|20)?\d\d\b", "[DATE]", s, "date")

    # (7) years -> relative to fiscal year T ([YEAR_T], [YEAR_T-1], [YEAR_T+1], ...); see year_token
    #   abbreviated: 'FY08' / 'FY 08' -> 'FY[YEAR_T-4]';  "'08" / "’08" -> year ('50-'99 -> 19xx, '00-'49 -> 20xx)
    s = sub(r"\bFY\s?'?(\d{2})\b", lambda m: "FY" + year_token(2000 + int(m.group(1)), T), s, "year")
    s = sub(r"(?<![\w$])['’](\d{2})\b", lambda m: year_token(2000 + int(m.group(1)) if int(m.group(1)) < 50
                                                           else 1900 + int(m.group(1)), T), s, "year")
    #   four-digit years 1950-2029, but not amounts or quantities:
    #   not after '$' / digit / ',' / '.' ($2,010), not followed by ',d' / '.d' (2010.5),
    #   not followed by million / billion / thousand / units / shares / %
    s = sub(r"(?<![\$\d,.])\b(19[5-9]\d|20[0-2]\d)\b(?![,.]\d)(?!\s*(?:million|billion|thousand|units|shares|%))",
            lambda m: year_token(int(m.group(1)), T), s, "year")
    return s, counts


# ------------------------------------------------------------------ masking (NER pass)
# ORG names that are kept unmasked because they carry meaning rather than identity ("the FASB issued ..."):
# regulators, standard setters and accounting standards, exchanges, rating agencies, government bodies, and generic
# terms spaCy often tags as ORG ("the Company", "Board of Directors"). Compared in lower case, with or without "the".
NER_KEEP = {"sec", "securities and exchange commission", "fasb", "financial accounting standards board", "irs",
            "internal revenue service", "federal reserve", "fed", "nyse", "new york stock exchange", "nasdaq", "amex",
            "congress", "epa", "fda", "food and drug administration", "fcc", "ferc", "pcaob", "gaap", "u.s. gaap", "iasb",
            "eitf", "sfas", "fsp", "fin", "apb", "sab", "aicpa", "treasury", "u.s. treasury", "department of defense",
            "dod", "department of energy", "medicare", "medicaid", "cms", "opec", "naic", "fdic", "occ", "ftc",
            "department of justice", "doj", "board", "board of directors", "the board of directors", "company",
            "the company", "audit committee", "compensation committee", "libor", "s&p", "standard & poor's", "moody's",
            "fitch", "otc", "ifrs", "sarbanes-oxley", "nasdaq global market", "nasdaq national market"}


def ner_mask(texts, n_process=16):
    """NER pass, run on the rule-masked chunks (after chunking). Catches names the rules miss: brand names, companies
    written without a legal suffix, people without a title.

    Only two spaCy entity types are replaced:  ORG -> [ORG],  PERSON -> [PERSON].
    All other labels (GPE, LOC, DATE, MONEY, PRODUCT, ...) are left as they are.

    An ORG / PERSON entity is skipped (kept unmasked) if
      - it is in NER_KEEP (with or without a leading 'the')
      - it already contains a mask token from the rule pass ('[COMPANY] Holdings')
      - it has no capital letter, or is a single character (spaCy false positives)
      - it is an accounting standard reference ('SFAS 123', 'FIN 48', 'ASC 820', ...)
    Returns the masked texts and the number of replacements per label.
    """
    import spacy
    # only the NER component is needed
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    out, n = [], {"ORG": 0, "PERSON": 0}
    for doc in nlp.pipe(texts, batch_size=64, n_process=n_process):
        s, last = [], 0   # rebuild the text: unchanged pieces between entities + mask tokens
        for e in doc.ents:
            if e.label_ not in n:
                continue  # not ORG / PERSON
            t = e.text.strip()
            low = re.sub(r"^the\s+", "", t.lower())
            if t.lower() in NER_KEEP or low in NER_KEEP or "[" in t or not re.search(r"[A-Z]", t) or len(t) < 2:
                continue
            if re.fullmatch(r"(SFAS|FIN|FSP|EITF|APB|SAB|SOP|ASC|FAS)\b.*", t):
                continue
            s.append(doc.text[last:e.start_char]); s.append(f"[{e.label_}]"); last = e.end_char; n[e.label_] += 1
        s.append(doc.text[last:])
        out.append("".join(s))
    return out, n


# ------------------------------------------------------------------ chunking
SEG = pysbd.Segmenter(language="en", clean=False)


def chunk_sentences(text, per=10, max_chunks=5, min_chars=250):
    sents = [x.strip() for x in SEG.segment(text) if x.strip()]
    chunks = [" ".join(sents[i:i + per]) for i in range(0, len(sents), per)][:max_chunks]
    return [c for c in chunks if len(c) >= min_chars]


# ------------------------------------------------------------------ prepare stage helpers
def load_meta():
    rows = []
    for y in YEARS:
        meta = pd.read_csv(f"{RAW}/ex/{y}.meta.txt", sep="\t", header=None, names=["key", "date", "url", "name_cik"],
                           dtype=str, quoting=3)
        pre = pd.read_csv(f"{RAW}/ex/{y}.logvol.-12.txt", sep=r"\s+", header=None, names=["prevol", "key"])
        post = pd.read_csv(f"{RAW}/ex/{y}.logvol.+12.txt", sep=r"\s+", header=None, names=["logvol", "key"])
        m = meta.merge(pre, on="key").merge(post, on="key")
        m["year"] = y
        rows.append(m)
    df = pd.concat(rows, ignore_index=True)
    # name_cik = 'ACME WIDGETS HOLDINGS INC /DE/ 1234567': EDGAR company name + CIK (SEC firm ID)
    df["cik"] = df["name_cik"].str.extract(r"(\d+)\s*$")[0]
    df["name"] = df["name_cik"].str.replace(r"\s*\d+\s*$", "", regex=True).str.strip()
    df["month"] = df["date"].str[4:6].astype(int)
    return df.drop(columns=["name_cik"])


def _read_year(args):
    """pass 1 worker: clean all MD&As of one filing year; returns {key: text} and word document counts."""
    y, keys = args
    cleaned, dfc = {}, collections.Counter()
    with tarfile.open(f"{RAW}/mda/{y}.mda.tgz") as tf:
        for m in tf:
            if not m.name.endswith(".mda"):
                continue
            key = os.path.basename(m.name)[:-4]
            if key not in keys:
                continue
            full = clean_mda(tf.extractfile(m).read().decode("utf-8", "ignore"))
            if len(full) >= 250:
                cleaned[key] = full
                dfc.update(set(re.findall(r"[a-z]+", full.lower())))
    return cleaned, dfc


_COMMON = None


def _init_worker(common):
    global _COMMON
    _COMMON = common


def _process_report(item):
    """pass 2 worker: fiscal year, rule masking (prefix only), chunking."""
    key, full, r = item
    T = detect_fiscal_year(full, r["year"], r["month"])
    prefix = full[:PREFIX_CHARS]
    masked, cnt = mask_text(prefix, name_variants(r["name"]), T, _COMMON)
    chunks = chunk_sentences(masked)
    if not chunks:
        return None, cnt
    raw_chunks = chunk_sentences(prefix)  # unmasked, only for the leak test
    return ({"key": key, "year": r["year"], "T": T, "cik": r["cik"], "name": r["name"], "prevol": r["prevol"],
             "logvol": r["logvol"], "chunks": chunks, "chunks_unmasked": raw_chunks[:len(chunks)]}, cnt)



def prepare():
    meta = load_meta().set_index("key")
    print("reports with meta + vol:", len(meta), meta.groupby("year").size().to_dict(), flush=True)
    cleaned, dfc = {}, collections.Counter()
    with Pool(len(YEARS)) as pool:
        for c, d in pool.map(_read_year, [(y, set(meta.index[meta.year == y])) for y in YEARS]):
            cleaned.update(c); dfc.update(d)
    # common words: lower-case words in > 1% of MD&As; one-word company-name variants among them are not masked
    common = {w for w, c in dfc.items() if c / len(cleaned) > 0.01}
    print("reports with >=250 chars:", len(cleaned), flush=True)
    items = [(k, full, {"year": int(meta.at[k, "year"]), "month": int(meta.at[k, "month"]), "name": meta.at[k, "name"],
                        "cik": meta.at[k, "cik"], "prevol": float(meta.at[k, "prevol"]),
                        "logvol": float(meta.at[k, "logvol"])}) for k, full in sorted(cleaned.items())]
    reports, tot = [], collections.Counter()
    with Pool(48, initializer=_init_worker, initargs=(common,)) as pool:
        for rec, cnt in pool.imap(_process_report, items, chunksize=20):
            tot.update(cnt)
            if rec is not None:
                reports.append(rec)
    rep = pd.DataFrame(reports)
    print("reports with >=1 chunk:", len(rep), rep.groupby("year").size().to_dict(),
          "| unique firms:", rep.cik.nunique(), "| chunks:", int(rep.chunks.str.len().sum()), flush=True)
    print("rule masking:", dict(tot), flush=True)
    # NER pass on all chunks
    flat = [c for cs in rep.chunks for c in cs]
    masked, n = ner_mask(flat)
    print("NER replacements:", n, flush=True)
    it = iter(masked)
    rep["chunks_rulemask"] = rep.chunks
    rep["chunks"] = [[next(it) for _ in cs] for cs in rep.chunks_rulemask]
    os.makedirs(OUT, exist_ok=True)
    rep.to_json(os.path.join(OUT, "reports.json"), orient="records", lines=True)
    json.dump({"years": YEARS, "rule_masking": dict(tot), "ner": n, "n_reports": len(rep)},
              open(os.path.join(OUT, "prep_info.json"), "w"), indent=2)


def split(a, D=OUT):
    rng = np.random.default_rng(a.seed)
    rep = pd.read_json(os.path.join(D, "reports.json"), lines=True, dtype={"cik": str})
    # 2,000 reports used in an earlier evaluation are kept out of the holdout, so no holdout text was evaluated before
    old = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "10k_excluded_holdout_keys.json")))["keys"])
    firms = np.array(sorted(set(rep.cik))); rng.shuffle(firms)
    elig = rep[~rep.key.isin(old)].groupby("cik").size()
    print(f"reports {len(rep)}, firms {rep.cik.nunique()}, reports excluded from the holdout {rep.key.isin(old).sum()}")
    hold_f, n, i = set(), 0, 0
    while n < a.n_hold:
        f = firms[i]; i += 1
        if elig.get(f, 0) == 0: firms = np.append(firms, f); continue  # keep for train/val
        hold_f.add(f); n += elig[f]
    rest = firms[i:]
    val_f, n = set(), 0
    nchunks = rep.groupby("cik").chunks.apply(lambda s: sum(len(c) for c in s))
    for f in rest:
        if n >= a.n_val: break
        val_f.add(f); n += nchunks[f]
    train_f = set(rep.cik) - hold_f - val_f

    def explode(r):
        return [{**{k: r[k] for k in COLS}, "chunk_idx": j, "text": c} for j, c in enumerate(r.chunks)]
    train = pd.DataFrame([x for _, r in rep[rep.cik.isin(train_f)].iterrows() for x in explode(r)])
    val = pd.DataFrame([x for _, r in rep[rep.cik.isin(val_f)].iterrows() for x in explode(r)])
    hrep = rep[rep.cik.isin(hold_f) & ~rep.key.isin(old)]
    hold = pd.DataFrame([{**{k: r[k] for k in COLS}, "chunk_idx": (j := int(rng.integers(len(r.chunks)))),
                          "text": r.chunks[j]} for _, r in hrep.iterrows()])

    tr_rep = train.drop_duplicates("key")
    ym = tr_rep.groupby("year").logvol.mean(); sd = float((tr_rep.logvol - tr_rep.year.map(ym)).std())
    pm = tr_rep.groupby("year").prevol.mean()
    for d in (train, val, hold):
        d["y"] = (d.logvol - d.year.map(ym)) / sd
        d["prevol_c"] = (d.prevol - d.year.map(pm)) / sd   # covariate on the same scale
    for name, d in [("train", train), ("val", val), ("holdout", hold)]:
        d.to_json(os.path.join(D, f"{name}.json"), orient="records", lines=True)
        print(f"{name}: {len(d)} chunks, {d.key.nunique()} reports, {d.cik.nunique()} firms, "
              f"years {d.drop_duplicates('key').year.value_counts().sort_index().to_dict()}")
    json.dump({"year_means_train": ym.to_dict(), "sd_train": sd, "seed": a.seed, "n_hold_firms": a.n_hold,
               "n_hold_firms_actual": len(hold_f), "n_val_firms": len(val_f)}, open(os.path.join(D, "split_info.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "prepare", "split"], default="all")
    ap.add_argument("--n_hold", type=int, default=2000, help="target # holdout reports")
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.stage in ("all", "prepare"):
        prepare()
    if a.stage in ("all", "split"):
        split(a)


if __name__ == "__main__":
    main()