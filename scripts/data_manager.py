#!/usr/bin/env python3

import csv
import gzip
import hashlib
from pathlib import Path
import sys
import urllib.request
from tqdm import tqdm
import pandas as pd

# Configuration – adjust only if the FTP location ever changes, or you wish to download extra properties

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras"
FILES = ("CID-Mass.gz", "CID-IUPAC.gz")
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'raw'


def _download(url: str, dest: str) -> None:
    """Download a file with a minimal progress bar."""
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := resp.read(8192):
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                sys.stderr.write(f"\r{dest}: {downloaded / total:6.1%}")
            else:
                sys.stderr.write(f"\r{dest}: {downloaded / 1024:6.1f} KiB")
            sys.stderr.flush()
    sys.stderr.write("\n")


def _md5_of_file(p: Path) -> str:
    """Return the hexadecimal MD5 digest of *p*."""
    h = hashlib.md5()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _verify_with_md5file(data_file: Path, md5_file: Path) -> bool:
    """Parse the .md5 file (<md5>  <filename>) and compare to local digest."""
    expected = md5_file.read_text().strip().split()[0].lower()
    observed = _md5_of_file(data_file)
    if expected == observed:
        print(f"{data_file.name} matches its .md5 checksum.")
        return True
    print(
        f"   {data_file.name} checksum mismatch!\n"
        f"    expected {expected}\n"
        f"    got      {observed}",
        file=sys.stderr,
    )
    return False


def _gunzip(src: Path, dest: Path) -> None:
    """Decompress src.gz → dest (plain text, line‑by‑line)."""
    with gzip.open(src, "rb") as fin, dest.open("wb") as f_out:
        while True:
            chunk = fin.read(8192)
            if not chunk:
                break
            f_out.write(chunk)


def ensure_file_present_and_unzipped():
    """
    Ensure raw files are downloaded, MD5-verified, and unzipped.
    """
    for file in FILES:
        gz_path = RAW_DATA_PATH / file
        unzipped_path = RAW_DATA_PATH / file.rstrip(".gz")
        md5_path = gz_path.with_suffix(".gz.md5")  # same as str(gz_path)+".md5"

        if unzipped_path.is_file():
            print(f"{unzipped_path.name} is present (unzipped).")
            continue  # already unzipped, nothing to do

        print(f"{unzipped_path.name} is not present.")
        print(f"Downloading {gz_path.name} and its .md5 …")
        _download(f"{BASE_URL}/{file}", str(gz_path))
        _download(f"{BASE_URL}/{file}.md5", str(md5_path))

        # verify MD5
        if not _verify_with_md5file(gz_path, md5_path):
            gz_path.unlink(missing_ok=True)
            raise ValueError(f"MD5 checksum mismatch for {gz_path.name}")

        # unzip once
        print(f"Unzipping {gz_path.name} → {unzipped_path.name} …")
        _gunzip(gz_path, unzipped_path)

        # optionally remove the MD5 file
        if md5_path.exists():
            md5_path.unlink()

        if gz_path.exists():
            gz_path.unlink()

def get_merged_slice(cids: set) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['cid', 'iupac', 'mass'] for the given CIDs.
    Uses unzipped files if available, otherwise downloads, verifies, and unzips once.
    """
    ensure_file_present_and_unzipped()

    cids = set(str(cid) for cid in cids)  # ensure string matching
    merged_data = {}  # key=cid, value=dict with cid/mass/iupac

    # process unzipped CID-Mass.tsv: CID in column 0, mass in column 3
    mass_file = RAW_DATA_PATH / "CID-Mass"
    print(f"Opening {mass_file}...")
    with open(mass_file, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc=f"Processing {mass_file}..."):
            cid = row[0]
            if cid in cids:
                merged_data.setdefault(cid, {"cid": cid, "mass": None, "iupac": None})
                merged_data[cid]["mass"] = row[3]

    # process unzipped CID-IUPAC.tsv: CID in column 0, iupac in column 1
    iupac_file = RAW_DATA_PATH / "CID-IUPAC"
    print(f"Opening {iupac_file}...")
    with open(iupac_file, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc=f"Processing {iupac_file}..."):
            cid = row[0]
            if cid in cids:
                merged_data.setdefault(cid, {"cid": cid, "mass": None, "iupac": None})
                merged_data[cid]["iupac"] = row[1]

    # convert merged dicts to DataFrame in desired order
    df = pd.DataFrame(merged_data.values(), columns=["cid", "iupac", "mass"])
    return df


# This is too expensive to load at once, blows up lower memory machines
def get_merged_slice_alt(cids: set) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['cid', 'iupac', 'mass'] for the given CIDs.
    Uses unzipped files if available, otherwise downloads, verifies, and unzips once.
    """
    ensure_file_present_and_unzipped()

    cids = set(str(cid) for cid in cids)  # ensure string matching

    # process unzipped CID-Mass.tsv: CID in column 0, mass in column 3
    mass_file = RAW_DATA_PATH / "CID-Mass"
    mass_df = pd.read_csv(mass_file, header=None, names=['cid', 'x', 'y', 'mass'], index_col='cid')[['mass']]
    mass_df = mass_df[mass_df.index.isin(cids)].copy()

    # process unzipped CID-IUPAC.tsv: CID in column 0, iupac in column 1
    iupac_file = RAW_DATA_PATH / "CID-IUPAC"
    iupac_df = pd.read_csv(iupac_file, header=None, names=['cid', 'iupac'], index_col='cid')[['iupac']]
    iupac_df = iupac_df[iupac_df.index.isin(cids)].copy()

    merged = mass_df.merge(iupac_df, on='cid', how='inner')

    return merged

