import gzip, io, tempfile, requests, pandas as pd
from pathlib import Path
import time
from datetime import datetime, timezone, date
from typing import List, Literal
import keyring
import numpy as np
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
    # Add more missing value patterns for visibility
    return _split_numeric(code, 1, miss=("9999", "99999", "+9999", "+99999", "999999", "016093"))  # metres


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

        self.KEEP_COLS = [
            "STATION", "NAME", "DATE",          # id / metadata
            "TMP", "DEW", "SLP", "WND", "VIS",
            "CIG", "LATITUDE", "LONGITUDE", "ELEVATION", "MA1"
        ]

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle duplicate columns and columns with .1, .2, etc. suffixes
        # Get a list of columns with their positions
        cols_with_position = [(i, col) for i, col in enumerate(df.columns)]
        # Create a dictionary to track the first occurrence of each base column name
        first_occurrence = {}
        # Create a list of column indices to keep
        indices_to_keep = []

        for i, col in cols_with_position:
            # Check if the column name has a .N suffix (like DEW.1)
            import re
            base_col = re.sub(r'\.\d+$', '', col)

            # If this is the first occurrence of the base column name, keep it
            if base_col not in first_occurrence:
                first_occurrence[base_col] = i
                indices_to_keep.append(i)

        # Keep only the first occurrence of each base column
        df = df.iloc[:, indices_to_keep]

        # --------- decode coded groups into new numeric cols ----------
        df["temp_c"] = df["TMP"].map(_parse_tmp)
        if "DEW" in df.columns:
            df["dew_c"] = df["DEW"].map(_parse_dew)
        # df["slp_hpa"] = df["SLP"].map(_parse_slp)
        df["vis_m"] = df["VIS"].map(_parse_vis)
        df["ceil_m"] = df["CIG"].map(_parse_cig)

        # Create empty columns for wind data if WND column doesn't exist or parsing fails
        if "WND" in df.columns:
            try:
                # Apply the parsing function to each row and create a DataFrame
                wnd_data = pd.DataFrame([_parse_wnd(x) for x in df["WND"]])
                # Add the parsed columns to the original DataFrame
                df["wind_dir_deg"] = wnd_data["wind_dir_deg"]
                df["wind_spd_ms"] = wnd_data["wind_spd_ms"]
            except Exception as e:
                print(f"Error parsing WND column: {e}")
                df["wind_dir_deg"] = np.nan
                df["wind_spd_ms"] = np.nan

        # Create empty columns for MA1 data if MA1 column doesn't exist or parsing fails
        if "MA1" in df.columns:
            try:
                # Apply the parsing function to each row and create a DataFrame
                ma1_data = pd.DataFrame([_parse_ma1(x) for x in df["MA1"]])
                # Add the parsed columns to the original DataFrame
                df["altim_hpa"] = ma1_data["altim_hpa"]
                df["stn_p_hpa"] = ma1_data["stn_p_hpa"]
            except Exception as e:
                print(f"Error parsing MA1 column: {e}")
                df["altim_hpa"] = np.nan
                df["stn_p_hpa"] = np.nan

        # --------- replace global sentinels in numeric cols -----------
        NUM_COLS = ["temp_c", "dew_c", "slp_hpa",
                    "vis_m", "ceil_m",
                    "LATITUDE", "LONGITUDE", "ELEVATION"]

        # Add columns only if they exist in the DataFrame
        for col in ["wind_dir_deg", "wind_spd_ms", "altim_hpa", "stn_p_hpa"]:
            if col in df.columns:
                NUM_COLS.append(col)

        # Only replace values in columns that exist in the DataFrame
        existing_cols = [col for col in NUM_COLS if col in df.columns]
        if existing_cols:
            df[existing_cols] = df[existing_cols].replace(
                {9999: np.nan, 99999: np.nan, -9999: np.nan,
                "+9999": np.nan, "+99999": np.nan, 999999: np.nan}
            )

        # Additional cleaning for visibility column
        if 'vis_m' in df.columns:
            # Replace any value that contains 9999 in the visibility column
            df['vis_m'] = df['vis_m'].apply(lambda x: np.nan if pd.notna(x) and '9999' in str(x) else x)

        # safe cast & interpolate
        for col in NUM_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        # Only interpolate columns that exist in the DataFrame
        if existing_cols:
            # For visibility column, use a more conservative interpolation approach
            if 'vis_m' in existing_cols:
                # First, interpolate all columns except visibility
                other_cols = [col for col in existing_cols if col != 'vis_m']
                if other_cols:
                    df[other_cols] = df[other_cols].interpolate(limit_direction="both")

                # First, interpolate visibility with a higher limit to fill more gaps
                df['vis_m'] = df['vis_m'].interpolate(limit=5, limit_direction="both")

                # Then, use forward and backward fill to fill any remaining gaps
                # Use ffill() and bfill() instead of fillna(method='ffill') to avoid FutureWarning
                df['vis_m'] = df['vis_m'].ffill().bfill()

                # If there are still NaN values (e.g., all NaN in a section), fill with the mean or a default value
                if df['vis_m'].isna().any():
                    # Calculate mean of non-NaN values, or use a default value if all are NaN
                    vis_mean = df['vis_m'].mean()
                    if pd.isna(vis_mean):
                        # If all values are NaN, use a reasonable default value for visibility (10000 meters = 10km)
                        df['vis_m'] = df['vis_m'].fillna(10000.0)
                    else:
                        df['vis_m'] = df['vis_m'].fillna(vis_mean)
            else:
                # If visibility column doesn't exist, interpolate all columns
                df[existing_cols] = df[existing_cols].interpolate(limit_direction="both")

        # Keep only metadata columns and derived columns, removing the original data columns
        metadata_cols = ["STATION", "NAME", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION"]
        derived_cols = ["temp_c", "dew_c", "slp_hpa", "vis_m", "ceil_m", 
                        "wind_dir_deg", "wind_spd_ms", "altim_hpa", "stn_p_hpa"]

        # Filter to keep only columns that exist in the DataFrame
        cols_to_keep = [col for col in metadata_cols + derived_cols if col in df.columns]

        # Return DataFrame with only the columns we want to keep
        return df[cols_to_keep]

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

    def _fetch_year_data(self, station_id: str, start_dt: date, end_dt: date, dfs: list):
        """
        Fetch data for a specific year and append to the dfs list.

        Args:
            station_id: The station ID to fetch data for
            start_dt: The start date
            end_dt: The end date
            dfs: List to append the fetched DataFrame to
        """
        try:
            raw_csv = self.fetch_hourly(station_id, start_dt, end_dt)
            # Check if the raw CSV has any data (more than just the header)
            if raw_csv.count('\n') > 1:
                df = pd.read_csv(io.StringIO(raw_csv), dtype=str, low_memory=False)
                df_filtered = df.dropna(thresh=10)
                # Check if the required columns exist in the DataFrame
                if all(col in df.columns for col in self.KEEP_COLS):
                    df_filtered = df_filtered[self.KEEP_COLS]
                    if not df_filtered.empty:
                        dfs.append(df_filtered)
                else:
                    print(f"Warning: Not all required columns found in data for {start_dt} to {end_dt}")
            else:
                print(f"No data returned for {start_dt} to {end_dt}")
        except Exception as exc:
            print(f"⚠ Error fetching data for {start_dt} to {end_dt}: {exc}")

        # Be polite to NOAA
        time.sleep(0.2)

    def get_hourly(self, use_single_request=True, max_years_per_request=5):
        """
        Fetch hourly data for all stations.

        Args:
            use_single_request: If True, fetch all data for a station in a single request.
                               If False, fetch data year by year.
            max_years_per_request: Maximum number of years to include in a single request.
                                  Only used if use_single_request is True.
        """
        # Use yesterday as the end date to avoid fetching future data
        today = date.today() - pd.Timedelta(days=1)

        for sid, row in tqdm(self.co_stns.set_index("id").iterrows(), total=len(self.co_stns)):
            start_ts = pd.Timestamp(self.START)
            begin_ts = row["begin"]
            first_day = max(start_ts, begin_ts).date()

            # Check if there's an existing CSV file for this station
            csv_path = self.OUTDIR / f"{sid}.csv"
            existing_data_found = False
            latest_date = None

            if csv_path.exists():
                try:
                    # Read the CSV file to determine the latest date in the dataset
                    existing_df = pd.read_csv(csv_path)
                    if 'DATE' in existing_df.columns and not existing_df.empty:
                        existing_df['DATE'] = pd.to_datetime(existing_df['DATE'])
                        latest_date = existing_df['DATE'].max()
                        # Use the day after the latest date as the start date for fetching new data
                        # This avoids re-fetching data we already have
                        new_start_day = (latest_date + pd.Timedelta(days=1)).date()
                        first_day = max(first_day, new_start_day)
                        existing_data_found = True
                        print(f"Found existing data for station {sid}. Latest date: {latest_date}")
                        print(f"Will fetch new data starting from: {first_day}")
                except Exception as e:
                    print(f"Error reading existing CSV file for station {sid}: {e}")

            # Set the end date to the minimum of today and the station's end date
            last_day = today if pd.isna(row["end"]) else min(today, row["end"].date())

            # If the first day is after the last day, there's no data to fetch
            if first_day > last_day:
                print(f"No new data to fetch for station {sid}. First day {first_day} is after last day {last_day}.")
                continue

            dfs = []

            if use_single_request:
                # Calculate total years in the request
                total_years = last_day.year - first_day.year + 1

                # If the date range is too large, split into chunks
                if total_years > max_years_per_request:
                    # Process in chunks of max_years_per_request
                    for chunk_start_year in tqdm(range(first_day.year, last_day.year + 1, max_years_per_request)):
                        chunk_end_year = min(chunk_start_year + max_years_per_request - 1, last_day.year)

                        chunk_start = date(chunk_start_year, 1, 1) if chunk_start_year > first_day.year else first_day
                        # If this is the current year, use yesterday's date as the end date
                        if chunk_end_year == date.today().year:
                            chunk_end = min(date.today() - pd.Timedelta(days=1), last_day)
                        else:
                            chunk_end = date(chunk_end_year, 12, 31) if chunk_end_year < last_day.year else last_day

                        try:
                            raw_csv = self.fetch_hourly(sid, chunk_start, chunk_end)
                            # Check if the raw CSV has any data (more than just the header)
                            if raw_csv.count('\n') > 1:
                                df = pd.read_csv(io.StringIO(raw_csv), dtype=str, low_memory=False)
                                df_filtered = df.dropna(thresh=10)
                                # Check if the required columns exist in the DataFrame
                                if all(col in df.columns for col in self.KEEP_COLS):
                                    df_filtered = df_filtered[self.KEEP_COLS]
                                    if not df_filtered.empty:
                                        dfs.append(df_filtered)
                                else:
                                    print(f"Warning: Not all required columns found in data for {chunk_start} to {chunk_end}")
                            else:
                                print(f"No data returned for {chunk_start} to {chunk_end}")
                        except Exception as exc:
                            print(f"⚠ Error fetching chunk {chunk_start} to {chunk_end}: {exc}")
                            print("Falling back to year-by-year fetching for this chunk")

                            # Fall back to year-by-year for this chunk
                            for year in range(chunk_start_year, chunk_end_year + 1):
                                y_start = date(year, 1, 1) if year > first_day.year else first_day
                                # If this is the current year, use yesterday's date as the end date
                                if year == date.today().year:
                                    y_end = min(date.today() - pd.Timedelta(days=1), last_day)
                                else:
                                    y_end = date(year, 12, 31) if year < last_day.year else last_day
                                self._fetch_year_data(sid, y_start, y_end, dfs)

                        # Be polite to NOAA
                        time.sleep(0.5)
                else:
                    # Single request for the entire date range
                    print(f"Fetching data from {first_day} to {last_day} for station {sid}")
                    try:
                        raw_csv = self.fetch_hourly(sid, first_day, last_day)
                        # Check if the raw CSV has any data (more than just the header)
                        if raw_csv.count('\n') > 1:
                            df = pd.read_csv(io.StringIO(raw_csv), dtype=str, low_memory=False)
                            df_filtered = df.dropna(thresh=10)
                            # Check if the required columns exist in the DataFrame
                            if all(col in df.columns for col in self.KEEP_COLS):
                                df_filtered = df_filtered[self.KEEP_COLS]
                                if not df_filtered.empty:
                                    dfs.append(df_filtered)
                            else:
                                print(f"Warning: Not all required columns found in data for {first_day} to {last_day}")
                        else:
                            print(f"No data returned for {first_day} to {last_day}")
                    except Exception as exc:
                        print(f"⚠ Error fetching data: {exc}")
                        print("Falling back to year-by-year fetching")

                        # Fall back to year-by-year approach
                        year = first_day.year
                        while year <= last_day.year:
                            y_start = date(year, 1, 1) if year > first_day.year else first_day
                            # If this is the current year, use yesterday's date as the end date
                            if year == date.today().year:
                                y_end = min(date.today() - pd.Timedelta(days=1), last_day)
                            else:
                                y_end = date(year, 12, 31) if year < last_day.year else last_day
                            self._fetch_year_data(sid, y_start, y_end, dfs)
                            year += 1
            else:
                # Original year-by-year approach
                year = first_day.year
                while year <= last_day.year:
                    y_start = date(year, 1, 1) if year > first_day.year else first_day
                    # If this is the current year, use yesterday's date as the end date
                    if year == date.today().year:
                        y_end = min(date.today() - pd.Timedelta(days=1), last_day)
                    else:
                        y_end = date(year, 12, 31) if year < last_day.year else last_day
                    self._fetch_year_data(sid, y_start, y_end, dfs)
                    year += 1
            if dfs:
                full_df = pd.concat(dfs, ignore_index=True)

                full_df = self._clean_df(full_df)

                # print(full_df.describe())

                full_df.to_csv(self.OUTDIR / f"{sid}.csv", index=False)


if __name__ == "__main__":
    noaa = NOAA()
    noaa.get_station_ids()

    # For testing, just use the first station
    # noaa.co_stns = noaa.co_stns.iloc[:2]
    print(noaa.co_stns)

    # Option 1: Use a single request for all years (default)
    # This reduces the number of API calls but may fail for very large date ranges
    noaa.get_hourly(use_single_request=True, max_years_per_request=5)

    # Option 2: Use year-by-year requests
    # This makes more API calls but is more reliable for large date ranges
    # noaa.get_hourly(use_single_request=False)

    # Option 3: Use chunked requests with a custom chunk size
    # This is a compromise between options 1 and 2
    # noaa.get_hourly(use_single_request=True, max_years_per_request=3)
