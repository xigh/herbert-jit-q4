import re, unicodedata
def slug(titre):
    s = unicodedata.normalize("NFKD", titre.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")
