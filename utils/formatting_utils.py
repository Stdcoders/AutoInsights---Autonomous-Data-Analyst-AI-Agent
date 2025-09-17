# utils/formatting.py
import csv
import os
import io
import tempfile
import chardet
import pandas as pd
import re
import google.generativeai as genai
from typing import Tuple, Optional

# configure genai only if you plan to use LLM fallback
import os as _os
if _os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=_os.getenv("GOOGLE_API_KEY"))

def detect_encoding(path: str, nbytes: int = 20000) -> str:
    with open(path, "rb") as f:
        raw = f.read(nbytes)
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    return enc

def try_read_csv(path: str, encoding: str, **kwargs) -> pd.DataFrame:
    # wrapper to try reading with pandas with several param combos
    # on_bad_lines='skip' used to be error_bad_lines
    read_attempts = [
        {"sep": None, "engine": "python"},  # let python engine sniff sep
        {"sep": ",", "engine": "c"},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
        {"sep": "|", "engine": "python"},
    ]
    for params in read_attempts:
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False, on_bad_lines="warn", **params, **kwargs)
            # if success and not empty, return
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    # final attempt: read everything as one column then split heuristically
    try:
        text = open(path, "r", encoding=encoding, errors="replace").read()
        # try to replace weird separators by comma if consistent
        guessed_sep = guess_delimiter(text)
        if guessed_sep:
            df = pd.read_csv(path, encoding=encoding, sep=guessed_sep, engine="python", low_memory=False, on_bad_lines="warn")
            if df is not None and not df.empty:
                return df
    except Exception:
        pass
    raise ValueError("Could not read CSV with automated readers")

def guess_delimiter(sample_text: str) -> Optional[str]:
    candidates = [",", ";", "\t", "|"]
    lines = sample_text.splitlines()[:10]
    best = None
    best_score = 0
    for sep in candidates:
        counts = [line.count(sep) for line in lines if line]
        if not counts:
            continue
        # prefer delimiter with consistent count across lines
        score = (sum(1 for c in counts if c == counts[0]) / len(counts)) * (sum(counts)/len(counts))
        if score > best_score:
            best_score = score
            best = sep
    return best

def clean_header_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # If columns look like 'Unnamed' or duplicate, try to fix by using first non-empty row as header
    if any(col.startswith("Unnamed") for col in df.columns) or len(set(df.columns)) < len(df.columns):
        # find first row that looks like header (many strings)
        for i in range(0, min(5, len(df))):
            row = df.iloc[i].astype(str).tolist()
            # pretty naive: if more than half entries are non-numeric, treat as header
            non_numeric = sum(1 for v in row if not re.match(r"^-?\d+(\.\d+)?$", v))
            if non_numeric >= len(row) / 2:
                new_header = [str(v).strip() for v in row]
                df = df.iloc[i+1:].copy()
                df.columns = new_header
                break
    # dedupe columns
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = list(cols[cols == dup].index)
        for idx_i, idx in enumerate(dup_idx[1:], start=1):
            cols[idx] = f"{cols[idx]}_{idx_i}"
    df.columns = cols
    return df

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace in string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
        # replace explicit 'nan' strings with real NaN
        df[c].replace({"nan": pd.NA, "": pd.NA}, inplace=True)
    # try to parse date-like columns
    for c in df.columns:
        if "date" in c.lower() or re.search(r"^d(ate)?", c.lower()):
            try:
                # Use infer_datetime_format to reduce parsing warnings
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            except Exception:
                pass
    return df

def llm_repair_csv(sample_text: str, max_return_chars:int=20000, model: str="gemini-1.5-flash") -> str:
    """Ask LLM to repair a broken CSV-ish text and return valid CSV text only."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY required for LLM repair")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model)
    prompt = f"""
You are a data engineer. I will provide a fragment of a CSV-like file that may have delimiter/header/quoting/encoding issues.
Your task: return a corrected CSV text (including header row) only — no explanation, no markdown fences.
If you cannot fully recover columns, extract as many columns and rows as possible and output them as CSV.
Input fragment (first ~2000 chars):
{sample_text[:2000]}
"""
    response = llm.generate_content(prompt, generation_config={"temperature":0.0,"max_output_tokens":800})
    text = response.candidates[0].content.parts[0].text
    # try to extract CSV from response (strip any surrounding commentary)
    match = re.search(r"(^.*\n.*\n.*)", text, re.DOTALL)
    # we will just return full text; caller will try to parse
    return text

def format_dataset(path: str, file_type: str="csv") -> str:
    """
    Try to auto-fix common formatting issues and return path to cleaned file.
    If no changes required, returns original path.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    enc = detect_encoding(path)
    if file_type.lower() == "csv":
        # 1) quick try: normal read
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            # if read succeeds, sanitise and return original file
            df = sanitize_df(df)
            # write sanitized to temp file and return that (so ingestion uses normalized file)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            return tmp.name
        except Exception:
            pass

        # 2) try smarter reader attempts
        try:
            df = try_read_csv(path, encoding=enc)
            df = clean_header_duplicates(df)
            df = sanitize_df(df)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            return tmp.name
        except Exception:
            pass

        # 3) last resort: ask LLM to repair file
        try:
            raw = open(path, "r", encoding=enc, errors="replace").read()
            repaired = llm_repair_csv(raw)
            # save repaired to tmp
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
            tmp.write(repaired)
            tmp.flush()
            tmp.close()
            # verify we can read repaired
            df = pd.read_csv(tmp.name, low_memory=False)
            df = sanitize_df(df)
            return tmp.name
        except Exception as e:
            raise RuntimeError(f"Could not auto-repair CSV: {e}")

    else:
        # for non-csv types we just return original
        return path
