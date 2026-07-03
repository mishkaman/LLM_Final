import re


def _clean(text):
    """Lowercase, drop punctuation/currency, collapse whitespace."""
    s = str(text).lower().strip()
    s = s.replace("$", " ").replace(",", "")
    s = re.sub(r"[^a-z0-9. ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _as_number(s):
    """Return float(s) if s is a plain number, else None."""
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(predicted, expected, acceptable_answers=None):
    """True if the expected answer (or an acceptable one) shows up in prediction."""
    candidates = [str(expected)] + [str(a) for a in (acceptable_answers or [])]
    p = _clean(predicted)

    for cand in candidates:
        c = _clean(cand)
        if not c:
            continue
        cn = _as_number(c)
        if cn is not None:
            # numeric answer: look for the same number anywhere in the prediction
            for tok in p.split():
                pn = _as_number(tok)
                if pn is not None and pn == cn:
                    return True
        else:
            # text answer: simple substring containment
            if c in p:
                return True
    return False


def answers_match(a, b):
    """Symmetric-ish equivalence test used for consensus / majority voting."""
    return is_correct(a, b) or is_correct(b, a)
