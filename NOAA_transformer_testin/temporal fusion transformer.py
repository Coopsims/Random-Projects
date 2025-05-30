import gzip, io, tempfile, requests, pandas as pd
from pathlib import Path
import time
from datetime import datetime, timezone, date
from typing import List, Literal
import keyring
from isd import Batch
from tqdm import tqdm

class NOAA:
    def __init__(self):
        self.STATE   = "CO"
        self.START   = date(2000, 1, 1)
        self.END     = date.today()
        self.OUTDIR  = Path("ghcnh_hourly_CO")
        self.OUTDIR.mkdir(exist_ok=True)
        self.ADS_BASE_URL     = "https://www.ncei.noaa.gov/access/services/data/v1"
        # Columns we actually care about (ADS “dataTypes” values)
        # Add/remove as you wish; STATION & DATE come for free.
        self.KEEP_COLS = [
            "STATION", "NAME", "DATE",          # id / metadata
            "TMP", "DEW", "SLP", "WND", "VIS",
            "CIG", "LATITUDE", "LONGITUDE", "ELEVATION", "DEW", "MA1"
        ]

    def get_station_ids(self):
        self.stations_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        stations = pd.read_csv(self.stations_url)

        co_stns = (
            stations
            .loc[(stations["CTRY"] == "US") & (stations["STATE"] == self.STATE)]
            .assign(
                id=lambda d: d.USAF.astype(str).str.zfill(6) + d.WBAN.astype(str).str.zfill(5),
                begin=lambda d: pd.to_datetime(d['BEGIN'], format='%Y%m%d', errors='coerce'),
                end=lambda d: pd.to_datetime(d['END'],   format='%Y%m%d', errors='coerce')
            )
            .loc[:, ['id','STATION NAME','begin','end']]
        )

        cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
        recent_mask = co_stns['end'].isna() | (co_stns['end'] >= cutoff_date)
        self.co_stns = co_stns[recent_mask].reset_index(drop=True)

        self.co_stns.to_csv("colorado_station_inventory.csv", index=False)
        print(f"{len(self.co_stns)} stations found")

    def fetch_hourly(self, station_id: str, start_dt: date, end_dt: date, retries: int = 3) -> str:
        """Return CSV text for GLOBAL_HOURLY rows between start_dt and end_dt."""
        params = {
            "dataset": "global-hourly",
            "stations": station_id,
            "startDate": start_dt.isoformat(),
            "endDate": end_dt.isoformat(),
            "units": "metric",
            "includeStationName": "true",
            "format": "csv",
            # Only request the columns we need
            "dataTypes": ",".join(self.KEEP_COLS),
        }
        for _ in range(retries):
            resp = requests.get(self.ADS_BASE_URL, params=params, timeout=60)
            if resp.ok:
                return resp.text
            time.sleep(2)
        raise RuntimeError(f"Request failed for {station_id} {start_dt}–{end_dt}: {resp.status_code}")

    def get_hourly(self):
        today = date.today()

        for sid, row in tqdm(self.co_stns.set_index("id").iterrows(), total=len(self.co_stns)):
            first_day = max(self.START, row["begin"]).date()          # obey 2000-01-01 floor
            last_day  = today if pd.isna(row["end"]) else min(today, row["end"].date())

            dfs = []                      # collect DataFrames, not CSV strings
            year = first_day.year
            while year <= last_day.year:
                y_start = date(year, 1, 1) if year > first_day.year else first_day
                y_end   = date(year, 12, 31) if year < last_day.year else last_day
                try:
                    raw_csv = self.fetch_hourly(sid, y_start, y_end)
                    # ------------------------------------------------------------------
                    # Keep only "complete" rows: at least 10 non‑missing fields
                    # ------------------------------------------------------------------
                    # Read everything as string, keep memory low
                    df = pd.read_csv(io.StringIO(raw_csv), dtype=str, low_memory=False)
                    df_filtered = df.dropna(thresh=10)          # tweak threshold if needed
                    df_filtered = df_filtered[self.KEEP_COLS]
                    if not df_filtered.empty:
                        dfs.append(df_filtered)
                except Exception as exc:
                    print(f"⚠ {exc}")
                year += 1
                time.sleep(0.2)        # be polite to NOAA
            if dfs:
                full_df = pd.concat(dfs, ignore_index=True)
                # -----------------------------------------------------------
                #  Save the cleaned data
                # -----------------------------------------------------------
                full_df.to_csv(self.OUTDIR / f"{sid}.csv", index=False)
                # -----------------------------------------------------------
                #  Save per‑column null counts for quick diagnostics
                # -----------------------------------------------------------
                null_counts = full_df.isna().sum()
                null_counts.to_csv(self.OUTDIR / f"{sid}_null_counts.csv",
                                   header=["null_count"])

if __name__ == "__main__":
    noaa = NOAA()
    noaa.get_station_ids()
    noaa.co_stns = noaa.co_stns.iloc[[0]]
    print(noaa.co_stns)
    noaa.get_hourly()

