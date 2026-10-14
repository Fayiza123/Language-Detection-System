"""Build a labelled sentence-level dataset from the NLTK UDHR corpus
(Universal Declaration of Human Rights in many languages)."""
import re, csv, os, random

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RAW = os.path.join(BASE, "udhr")
LANGS = {
    "English": ("English-Latin1", "latin-1"),
    "French": ("French_Francais-Latin1", "latin-1"),
    "German": ("German_Deutsch-Latin1", "latin-1"),
    "Spanish": ("Spanish_Espanol-Latin1", "latin-1"),
    "Italian": ("Italian_Italiano-Latin1", "latin-1"),
    "Portuguese": ("Portuguese_Portugues-Latin1", "latin-1"),
    "Dutch": ("Dutch_Nederlands-Latin1", "latin-1"),
    "Swedish": ("Swedish_Svenska-Latin1", "latin-1"),
    "Polish": ("Polish_Polski-Latin2", "iso8859-2"),
    "Turkish": ("Turkish_Turkce-UTF8", "utf-8-sig"),
    "Russian": ("Russian_Russky-UTF8", "utf-8-sig"),
    "Greek": ("Greek_Ellinika-UTF8", "utf-8-sig"),
    "Hindi": ("Hindi-UTF8", "utf-8-sig"),
    "Bengali": ("Bengali-UTF8", "utf-8-sig"),
    "Tamil": ("Tamil-UTF8", "utf-8-sig"),
    "Kannada": ("Kannada-UTF8", "utf-8-sig"),
}

def segments(text, min_len=25, max_len=140):
    """Split on sentence punctuation, then merge pieces into 25+ char samples."""
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.;:!?,\u0964])\s+", text)
    out, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip()
        if len(buf) >= min_len:
            out.append(buf[: max_len * 2])
            buf = ""
    return out

def main():
    rows = []
    for lang, (fn, enc) in LANGS.items():
        with open(os.path.join(RAW, fn), encoding=enc, errors="ignore") as f:
            segs = segments(f.read())
        rows += [(s, lang) for s in segs]
        print(f"{lang:11s} {len(segs)} samples")
    random.Random(42).shuffle(rows)
    out = os.path.join(BASE, "dataset.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["text", "language"]); w.writerows(rows)
    print("Total:", len(rows), "->", out)

if __name__ == "__main__":
    main()
