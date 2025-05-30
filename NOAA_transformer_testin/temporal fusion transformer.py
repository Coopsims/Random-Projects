import gzip, io, tempfile, requests, pandas as pd
from pathlib import Path
import time
from datetime import datetime, timezone, date
from typing import List, Literal
import keyring
from isd import Batch
from tqdm import tqdm


def _split_numeric(raw, scale=1, miss=("9999", "99999", "+9999", "+99999")):
    v = str(raw).split(",")[0].lstrip("+")
    return pd.NA if v in miss else float(v) / scale


def _parse_tmp(code):
    return _split_numeric(code, 10)


def _parse_dew(code):
    return _split_numeric(code, 10)


def _parse_slp(code):
    return _split_numeric(code, 10)


def _parse_vis(code):
    return _split_numeric(code, 1)  # metres


def _parse_cig(code):  # feet→m
    h = _split_numeric(code, 1)
    return pd.NA if pd.isna(h) else h * 0.3048


def _parse_wnd(code):
    if pd.isna(code): return pd.Series({"wind_dir_deg": pd.NA,
                                        "wind_spd_ms": pd.NA})
    d, _, _, s, _ = code.split(",")
    dir_ = pd.NA if d.startswith("99") else float(d)
    spd = _split_numeric(s, 10, miss=("9999",))
    return pd.Series({"wind_dir_deg": dir_, "wind_spd_ms": spd})


def _parse_ma1(code):
    if pd.isna(code): return pd.Series({"altim_hpa": pd.NA,
                                        "stn_p_hpa": pd.NA})
    alt, _, stn, _ = code.split(",")
    return pd.Series({"altim_hpa": _split_numeric(alt, 10),
                      "stn_p_hpa": _split_numeric(stn, 10)})

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
            "CIG", "LATITUDE", "LONGITUDE", "ELEVATION", "MA1"
        ]

    ####################################################################
    # 2)  rewrite _clean_df  (replace the whole old body)
    ####################################################################
    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # --------- decode coded groups into new numeric cols ----------
        df["temp_c"] = df["TMP"].map(_parse_tmp)
        df["dew_c"] = df["DEW"].map(_parse_dew)
        df["slp_hpa"] = df["SLP"].map(_parse_slp)
        df["vis_m"] = df["VIS"].map(_parse_vis)
        df["ceil_m"] = df["CIG"].map(_parse_cig)

        wnd_parsed = df["WND"].map(_parse_wnd)
        df = pd.concat([df, wnd_parsed], axis=1)

        ma1_parsed = df["MA1"].map(_parse_ma1)
        df = pd.concat([df, ma1_parsed], axis=1)

        # --------- replace global sentinels in numeric cols -----------
        NUM_COLS = ["temp_c", "dew_c", "slp_hpa",
                    "wind_dir_deg", "wind_spd_ms",
                    "vis_m", "ceil_m",
                    "altim_hpa", "stn_p_hpa",
                    "LATITUDE", "LONGITUDE", "ELEVATION"]

        df.replace({9999: pd.NA, 99999: pd.NA,
                    -9999: pd.NA, "+9999": pd.NA, "+99999": pd.NA},
                   inplace=True)

        # safe cast & interpolate
        for col in NUM_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        df[NUM_COLS] = df[NUM_COLS].interpolate(limit_direction="both")

        return df

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
            first_day = max(self.START, row["begin"]).date()
            last_day  = today if pd.isna(row["end"]) else min(today, row["end"].date())

            dfs = []
            year = first_day.year
            while year <= last_day.year:
                y_start = date(year, 1, 1) if year > first_day.year else first_day
                y_end   = date(year, 12, 31) if year < last_day.year else last_day
                try:
                    raw_csv = self.fetch_hourly(sid, y_start, y_end)

                    df = pd.read_csv(io.StringIO(raw_csv), dtype=str, low_memory=False)
                    df_filtered = df.dropna(thresh=10)
                    df_filtered = df_filtered[self.KEEP_COLS]
                    if not df_filtered.empty:
                        dfs.append(df_filtered)
                except Exception as exc:
                    print(f"⚠ {exc}")
                year += 1
                time.sleep(0.2)        # be polite to NOAA
            if dfs:
                full_df = pd.concat(dfs, ignore_index=True)

                full_df = self._clean_df(full_df)

                print(full_df.describe())

                full_df.to_csv(self.OUTDIR / f"{sid}.csv", index=False)


if __name__ == "__main__":
    noaa = NOAA()
    noaa.get_station_ids()
    noaa.co_stns = noaa.co_stns.iloc[[0]]
    print(noaa.co_stns)
    noaa.get_hourly()
