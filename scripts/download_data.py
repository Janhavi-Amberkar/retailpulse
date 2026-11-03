"""Download the public UCI Online Retail II dataset (CC BY 4.0) into data/raw/ (needed once)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retailpulse import config  # noqa: E402
from retailpulse.etl import extract  # noqa: E402

if __name__ == "__main__":
    if config.RAW_FILE.exists():
        print(f"already present: {config.RAW_FILE}")
    else:
        p = extract.download_raw()
        print(f"saved {p} ({p.stat().st_size / 1e6:.0f} MB). Source: {config.UCI_PAGE}")
