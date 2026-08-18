# -*- coding: utf-8 -*-
"""
pubchem_lookup.py

some pubchem API calls with ensured compliance (https://www.ncbi.nlm.nih.gov/home/about/policies/)

currently implemented mass lookups:
- lookup 1000 matching cids for a lot of smarts
- pick 
"""

from __future__ import annotations

import logging
import os
import time
import threading
import pytz

from datetime import datetime, time as dtime, timedelta
from datetime import timezone
from typing import List, Tuple, Optional


import pandas as pd
import requests
from tqdm.auto import tqdm
from urllib.parse import quote, urlencode
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Logging & NCBI contact information
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Fill these in (or export the env-vars before launching Python)
NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "your.email@domain.org")
NCBI_TOOL:  str = os.getenv("NCBI_TOOL", "pubchem_lookup")

if not NCBI_EMAIL or not NCBI_TOOL:
    raise RuntimeError(
        "You must provide both NCBI_EMAIL and NCBI_TOOL (via env vars or constants)."
    )

# Disclaimer (required by NCBI)
NCBI_DISCLAIMER = """
NCBI Disclaimer:
Data retrieved from PubChem are provided by the National Center for
Biotechnology Information (NCBI).  NCBI does not claim copyright on
PubMed abstracts; journal publishers or authors may hold those rights.
Please consult your legal counsel before redistributing copyrighted
material.
"""
log.info(NCBI_DISCLAIMER.strip())

# Rate limiter (thread-safe) - default max 3 QPS (NCBI policy)
class QPSRateLimiter:
    """Enforces a maximum number of requests per second across threads."""

    def __init__(self, max_qps: float = 3.0):
        if max_qps <= 0:
            raise ValueError("max_qps must be > 0")
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        """Block just long enough to keep the QPS budget."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_ts = time.time()


# Night-only / weekend enforcement (Eastern Time)
def _is_allowed_window() -> bool:
    """True if we are inside the NCBI-approved 21pm - 5am ET window (or weekend)."""
    now_utc = datetime.now(timezone.utc)

    eastern = pytz.timezone("America/New_York")
    now_et = now_utc.replace(tzinfo=pytz.utc).astimezone(eastern)

    # Weekends are always okay
    if now_et.weekday() >= 5:
        return True

    # Weekday night window: 21:00 - 05:00 ET
    start = dtime(21, 0)
    end = dtime(5, 0)
    if start <= now_et.time() or now_et.time() < end:
        return True
    return False


# noinspection PyUnboundLocalVariable
def _sleep_until_allowed() -> None:
    """Sleep until the next allowed window (tonight or upcoming weekend)."""
    now_utc = datetime.now(timezone.utc)

    eastern = pytz.timezone("America/New_York")
    now_et = now_utc.replace(tzinfo=pytz.utc).astimezone(eastern)

    # Compute next night start (21:00 ET)
    today = now_et.date()
    tonight_start = datetime.combine(today, dtime(21, 0))
    if now_et.time() >= dtime(21, 0):
        tonight_start += timedelta(days=1)

    # Compute next weekend start (Saturday 00:00 ET)
    days_until_sat = (5 - today.weekday()) % 7
    weekend_start = datetime.combine(
        today + timedelta(days=days_until_sat), dtime(0, 0)
    )

    # Choose the earlier one
    target = min(tonight_start, weekend_start)

    # Convert back to UTC for sleeping
    if "pytz" in globals():
        target_utc = target.astimezone(pytz.utc)
    else:
        target_utc = target - timedelta(hours=5)  # rough offset

    seconds = (target_utc - now_utc).total_seconds()
    seconds = max(seconds, 0)
    log.info(
        "Outside allowed window - sleeping %.0fs (~%s)...",
        seconds,
        str(timedelta(seconds=int(seconds))),
    )
    time.sleep(seconds)


def ensure_allowed_window(ignore_time_window: bool = False) -> None:
    """
    Block until the NCBI-approved time window is active,
    unless ``ignore_time_window`` is True (useful for debugging).
    """
    if ignore_time_window:
        return
    if not _is_allowed_window():
        _sleep_until_allowed()


# Robust GET with exponential back-off (used for every HTTP call)
def _robust_get(
    url: str,
    *,
    timeout: int = 60,
    max_retries: int = 3,
) -> requests.Response:
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            log.info("GET %s (attempt %d)", url, attempt)
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as exc:
            log.warning("Attempt %d failed: %s", attempt, exc)
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Unreachable in _robust_get")  # pragma: no cover


# Fast-substructure (SMARTS -> CID) query
def get_pubchem_matching_cids(
    smarts_pattern: str,
    *,
    limit: Optional[int] = 1000,
    strip_hydrogen: bool = True,
    timeout: int = 60,
    email: str = NCBI_EMAIL,
    tool: str = NCBI_TOOL,
) -> List[int]:
    """
    Return PubChem CIDs that match ``smarts_pattern``.
    The request includes the mandatory NCBI ``email`` and ``tool`` fields.
    """
    encoded = quote(smarts_pattern, safe="")
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/smarts"

    params = {
        "StripHydrogen": "True" if strip_hydrogen else "False",
        "email": email,
        "tool": tool,
    }
    if limit is not None:
        params["MaxRecords"] = str(limit)

    url = f"{base}/{encoded}/cids/TXT?{urlencode(params)}"

    resp = _robust_get(url, timeout=timeout)
    raw = resp.text.strip().splitlines()
    if not raw or raw == [""]:
        return []

    return [int(tok) for tok in raw if tok.isdigit()]


# Worker for the ThreadPoolExecutor (SMARTS -> CID)
def _cid_worker(
    smarts: str,
    limit: Optional[int],
    strip_hydrogen: bool,
    timeout: int,
    rate_limiter: Optional[QPSRateLimiter],
) -> Tuple[str, List[int]]:
    if rate_limiter is not None:
        rate_limiter.wait()
    return smarts, get_pubchem_matching_cids(
        smarts,
        limit=limit,
        strip_hydrogen=strip_hydrogen,
        timeout=timeout,
    )


# Public parallel lookup (returns original DF + CID list)
def parallel_cid_lookup(
    df: pd.DataFrame,
    smarts_col: str = "smarts",
    *,
    limit: Optional[int] = None,
    strip_hydrogen: bool = True,
    timeout: int = 60,
    max_workers: Optional[int] = None,
    max_qps: Optional[float] = 4.0,      # default = NCBI policy
    show_progress: bool = True,
    ignore_time_window: bool = False,
) -> pd.DataFrame:
    """
    Parallel lookup of PubChem CIDs for every SMARTS string in ``df[smarts_col]``.

    Returns ``df`` with two added columns:
        * ``cids``          - list of matching PubChem CIDs
    """
    # One global limiter for the whole job (covers both API endpoints)
    rate_limiter = QPSRateLimiter(max_qps) if max_qps is not None else None

    # Respect the night-only window *once* before launching threads
    ensure_allowed_window(ignore_time_window=ignore_time_window)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        worker = partial(
            _cid_worker,
            limit=limit,
            strip_hydrogen=strip_hydrogen,
            timeout=timeout,
            rate_limiter=rate_limiter,
        )
        iterator = executor.map(worker, df[smarts_col].astype(str))
        if show_progress:
            iterator = tqdm(
                iterator, total=len(df), desc="SMARTS -> CID", unit="smarts"
            )

        # each item is the (smarts, cids) tuple _cid_worker returns
        out_df = df.copy()
        out_df["cids"] = out_df[smarts_col].map(dict(iterator)).apply(
            lambda x: x if isinstance(x, list) else []
        )

    return out_df


def _fetch_properties_batch(
    batch: List[int],
    properties: List[str],
    *,
    timeout: int,
    email: str,
    tool: str,
) -> List[dict]:
    """
    Request PubChem PropertyTable for ``batch`` (max 1000CIDs per request
    - PubChem will happily accept longer lists, but 1000 keeps the URL short).
    Returns a list of property dictionaries.
    """
    # Build the query string with the mandatory NCBI parameters
    params = {"email": email, "tool": tool}
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{','.join(str(cid) for cid in batch)}/property/"
        f"{','.join(properties)}/JSON?{urlencode(params)}"
    )
    resp = _robust_get(url, timeout=timeout)
    data = resp.json()
    return data.get("PropertyTable", {}).get("Properties", [])



def get_compound_properties(
    cids: List[int],
    properties: List[str],
    *,
    batch_size: int = 1000,          # how many CIDs per property request
    timeout: int = 60,
    max_qps: float = 3.0,
    email: str = NCBI_EMAIL,
    tool: str = NCBI_TOOL,
    ignore_time_window: bool = False,
) -> pd.DataFrame:
    """
    Given a list of PubChem CIDs, return a dataframe with their compound properties

    Parameters
    ----------
    cids           - list of integer PubChem CIDs (duplicates are ignored)
    properties     - list of strings corresponding to compound properties to be fetched
    top_n          - number of lowest-MW hits to return (default 10)
    batch_size     - how many CIDs to request per HTTP call (default 1000)
    timeout, max_qps - same semantics as the rest of the module
    ignore_time_window - set True only for quick debugging; production should keep it False
    """
    if not cids:
        return pd.DataFrame(columns=["CID"] + properties)

    # Remove duplicates and sort for deterministic batching
    cids = sorted(set(cids))

    rate_limiter = QPSRateLimiter(max_qps)

    # Respect the night-only window *once* before the first batch
    ensure_allowed_window(ignore_time_window=ignore_time_window)

    all_props: List[dict] = []

    for i in range(0, len(cids), batch_size):
        batch = cids[i : i + batch_size]
        rate_limiter.wait()
        try:
            all_props.extend(_fetch_properties_batch(
                batch, properties=properties, timeout=timeout, email=email, tool=tool
            ))
        except Exception as exc:  # pragma: no cover - network errors are rare in tests
            log.warning("Failed to fetch properties for batch %s-%s: %s", i, i + batch_size, exc)

    cleaned = []
    for p in all_props:
        try:
            record = {"CID": int(p["CID"])}
            for prop in properties:
                # Use .get to avoid KeyError if the property is missing
                record[prop] = p.get(prop, None)
            cleaned.append(record)
        except ValueError:
            continue

    return pd.DataFrame(cleaned)


def get_lowest_mw_iupac_names(
    df,
    compounds_properties,
    n=10,
    cid_col="CID",
    mass_col="MolecularWeight",
    iupac_col="IUPACName",
):
    best_iupac_lists = []
    import ast

    for i, row in df.iterrows():
        cids_list = row['cids']
        # safely convert stringified lists
        if isinstance(cids_list, str):
            try:
                cids_list = ast.literal_eval(cids_list)
            except (ValueError, SyntaxError):
                cids_list = []

        if not cids_list:
            best_iupac_lists.append([])
            continue

        subset = compounds_properties[compounds_properties[cid_col].isin(cids_list)]
        subset_sorted = subset.sort_values(mass_col, ascending=True)

        # Get top_n IUPAC names
        top_names = subset_sorted[iupac_col].head(n).to_list()
        best_iupac_lists.append(top_names)

    return best_iupac_lists


# Convenience wrapper: SMARTS -> smallest-MW compounds (single call)
def lookup_smallest_mw_from_smarts(
    df: pd.DataFrame,
    smarts_col: str = "smarts",
    *,
    limit_per_smarts: Optional[int] = 1000,
    top_n: int = 10,
    batch_size: int = 100,
    max_qps: float = 3.0,
    timeout: int = 60,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    ignore_time_window: bool = False,
) -> pd.DataFrame:

    # SMARTS -> CID (parallel, with rate-limiting)
    smarts_to_cids_df = parallel_cid_lookup(
        df,
        smarts_col=smarts_col,
        limit=limit_per_smarts,
        max_qps=max_qps,
        timeout=timeout,
        max_workers=max_workers,
        show_progress=show_progress,
        ignore_time_window=ignore_time_window,
    )

    # Flatten all CIDs for bulk property fetch
    all_cids = sorted({cid for sublist in smarts_to_cids_df["cids"] for cid in sublist})

    # Fetch properties
    compound_df = get_compound_properties(
        all_cids,
        properties=['IUPACName', 'MolecularWeight'],
        batch_size=batch_size,
        timeout=timeout,
        max_qps=max_qps,
        ignore_time_window=ignore_time_window,
    )

    smarts_to_cids_df['best_IUPAC_names'] = get_lowest_mw_iupac_names(smarts_to_cids_df, compound_df, top_n)

    return smarts_to_cids_df


# Simple demo if file is executed directly
if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "smarts": [
                "O=[C]([R])[N]([R])[Nar]",
                "c1ccccc1",           # benzene ring
                "C(=O)N",             # simple amide
                "invalidSMARTS",      # will give [] for CIDs
            ]
        }
    )
    log.info("Running demo - one-off lookup")
    out = lookup_smallest_mw_from_smarts(
        demo,
        top_n=5,
        max_qps=3,                 # stay under NCBI limit
        ignore_time_window=True,   # set False for a real production run
    )
    print("\n=== SMARTS -> CIDs =====================")
    print(out[["smarts", "cids"]])
    print("\n=== Top-5 lowest-MW IUPAC names ===============")
    print(out[["smarts", "best_IUPAC_names"]])