from __future__ import annotations

import argparse
from contextlib import closing
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from urllib.parse import urlsplit
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIRST_EXPORT, LAST_EXPORT = "2021-01-01", "2026-01-11"
FIRST_WEEK, LAST_WEEK = "2021-01-04", "2026-01-05"
TRUTH_CUTOFF = "2025-01-06"  # Exclusive: count_2024 includes exports through Jan 5.
MISSING = ["2022-11-10", "2023-03-23"] + [
    str(day.date()) for day in pd.date_range("2025-06-14", "2025-07-01")
]
ROOT_CODES = ["03", "04", "09", "13", "14", "17", "18", "19"]
FIELDS = ["GLOBALEVENTID", "SQLDATE", "EventRootCode", "AvgTone",
          "GoldsteinScale", "SOURCEURL", "ActionGeo_CountryCode", "export_date"]
ACTOR = ["Code", "Name", "CountryCode", "KnownGroupCode", "EthnicCode",
         "Religion1Code", "Religion2Code", "Type1Code", "Type2Code", "Type3Code"]
GEO = ["Type", "FullName", "CountryCode", "ADM1Code", "Lat", "Long", "FeatureID"]
SCHEMA = (
    ["GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate"]
    + [f"Actor{actor}{field}" for actor in (1, 2) for field in ACTOR]
    + ["IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
       "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone"]
    + [f"{actor}Geo_{field}" for actor in ("Actor1", "Actor2", "Action") for field in GEO]
    + ["DATEADDED", "SOURCEURL", "export_date"]
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monday(value):
    value = pd.Timestamp(value).normalize()
    return value - pd.Timedelta(days=value.dayofweek)


def daily_files(raw, start=FIRST_EXPORT, end=LAST_EXPORT, missing=MISSING):
    missing = {pd.Timestamp(day) for day in missing}
    for date in pd.date_range(start, end):
        day = date.strftime("%Y%m%d")
        candidates = [Path(raw)/folder/"IS"/f"{day}.csv.gz"
                      for folder in ("gdelt", "spike_2025")]
        files = [path for path in candidates if path.exists()]
        if date in missing:
            if files:
                raise ValueError(f"Documented missing export now exists: {day}; review protocol")
            continue
        if len(files) != 1:
            raise ValueError(f"Expected one raw export for {day}, found {len(files)}")
        path = files[0]
        meta = json.loads(path.with_suffix(".json").read_text())
        if (meta.get("status") != "ok" or meta.get("country") != "IS"
                or str(meta.get("export_date")) != day
                or sha256(path) != meta.get("subset_sha256")):
            raise ValueError(f"Raw metadata/checksum mismatch: {path}")
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream)) != SCHEMA:
                raise ValueError(f"Wrong GDELT schema: {path}")
        frame = pd.read_csv(path, usecols=FIELDS, dtype=str, keep_default_na=False)
        if (len(frame) != meta["country_rows"] or not frame.export_date.eq(day).all()
                or not frame.ActionGeo_CountryCode.eq("IS").all()):
            raise ValueError(f"Raw country/date/row count mismatch: {path}")
        yield date, frame, meta


def unique_rows(frame, connection):
    keep = []
    columns = [name for name in FIELDS if name not in ("GLOBALEVENTID", "export_date")]
    for position, values in enumerate(frame[["GLOBALEVENTID", *columns]].itertuples(index=False, name=None)):
        event_id = int(values[0])
        fingerprint = hashlib.sha256(json.dumps(values[1:], ensure_ascii=False).encode()).digest()
        cursor = connection.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (event_id, fingerprint))
        if cursor.rowcount:
            keep.append(position)
        elif connection.execute("SELECT fingerprint FROM seen WHERE event_id=?", (event_id,)).fetchone()[0] != fingerprint:
            raise ValueError(f"Conflicting aggregation fields for event ID {event_id}")
    return frame.iloc[keep]


def empty_stats():
    return {"n": 0, "tone_sum": 0., "tone_n": 0, "gold_sum": 0.,
            "gold_sq": 0., "gold_n": 0, "domains": set(), "roots": {}}


def grace_truth(panel, missing, legacy_before=None):
    weeks = panel.index
    bad_weeks = {monday(day) for day in missing}
    truth = pd.Series(np.nan, index=weeks, name="count_7d_grace")
    for index, week in enumerate(weeks[:-1]):
        require_grace = legacy_before is None or week >= pd.Timestamp(legacy_before)
        if week not in bad_weeks and (not require_grace or weeks[index+1] not in bad_weeks):
            truth.iloc[index] = panel.iloc[index+1, index]
    return truth


def aggregate(daily, weeks, missing=MISSING, truth_cutoff=TRUTH_CUTOFF, legacy_grace_before=None):
    weeks = pd.DatetimeIndex(weeks, name="week_start")
    mapping = {week: i for i, week in enumerate(weeks)}
    increments = np.zeros((len(weeks), len(weeks)))
    final_counts = np.zeros(len(weeks))
    stats, grace_volumes = {}, {}
    audit = {"raw_rows": 0, "unique_rows": 0, "duplicate_rows": 0, "files": 0}
    download_dates = []
    # Disk-backed IDs bound RAM; scratch data are removed even on failure.
    with tempfile.TemporaryDirectory(prefix="forecast-preprocess-") as scratch:
        with closing(sqlite3.connect(str(Path(scratch)/"ids.sqlite"))) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE seen (event_id INTEGER PRIMARY KEY, fingerprint BLOB)")
            for date, frame, metadata in daily:
                date = pd.Timestamp(date)
                audit["raw_rows"] += len(frame)
                frame = unique_rows(frame, connection).copy()
                audit["unique_rows"] += len(frame)
                dates = pd.to_datetime(frame.SQLDATE, format="%Y%m%d", errors="raise")
                if (dates > date).any():
                    raise ValueError(f"Event date after export date: {date.date()}")
                if not frame.EventRootCode.isin([f"{i:02}" for i in range(1, 21)]).all():
                    raise ValueError(f"Invalid CAMEO root: {date.date()}")
                frame["week"] = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
                frame["conflict"] = frame.EventRootCode.isin(["18", "19"])
                report_index = max(0, weeks.searchsorted(monday(date)))
                if report_index >= len(weeks):
                    raise ValueError("Export extends beyond the requested calendar")
                for week, group in frame.groupby("week", sort=False):
                    conflict_count = int(group.conflict.sum())
                    if week in mapping:
                        index = mapping[week]
                        increments[report_index, index] += conflict_count
                        if date < pd.Timestamp(truth_cutoff):
                            final_counts[index] += conflict_count
                    if date < week + pd.Timedelta(days=14):
                        grace_volumes[week] = grace_volumes.get(week, 0) + len(group)
                    if week not in mapping or date >= week + pd.Timedelta(days=7):
                        continue
                    record = stats.setdefault(week, empty_stats())
                    tone = pd.to_numeric(group.AvgTone, errors="coerce").dropna()
                    gold = pd.to_numeric(group.GoldsteinScale, errors="coerce").dropna()
                    record["n"] += len(group)
                    record["tone_sum"] += float(tone.sum())
                    record["tone_n"] += len(tone)
                    record["gold_sum"] += float(gold.sum())
                    record["gold_sq"] += float((gold**2).sum())
                    record["gold_n"] += len(gold)
                    for url in group.SOURCEURL:
                        try:
                            record["domains"].add(urlsplit(url).hostname or "[missing/non-URL]")
                        except ValueError:
                            record["domains"].add("[malformed URL]")
                    for code, count in group.EventRootCode.value_counts().items():
                        record["roots"][code] = record["roots"].get(code, 0) + int(count)
                audit["files"] += 1
                download_dates.append(metadata["downloaded_at"])
                if audit["files"] % 150 == 0:
                    connection.commit()
                    print(f"Raw exports: {audit['files']}; unique records: {audit['unique_rows']:,}", flush=True)
    bad_weeks = {monday(day) for day in missing}
    panel = pd.DataFrame(increments.cumsum(axis=0), index=weeks, columns=weeks)
    panel.loc[:, panel.columns.isin(bad_weeks)] = np.nan
    records = []
    for index, week in enumerate(weeks):
        record = stats.get(week, empty_stats())
        n, total = record["gold_n"], record["n"]
        variance = max(0., (record["gold_sq"]-record["gold_sum"]**2/n)/(n-1)) if n > 1 else np.nan
        previous_week = week-pd.Timedelta(days=7)
        previous = grace_volumes.get(previous_week, 0)
        news = {
            "week_start": week,
            "forecast_issued_at_utc": week+pd.Timedelta(days=7, hours=12),
            "asof_export_before": week+pd.Timedelta(days=7),
            "tone_latest": record["tone_sum"]/record["tone_n"] if record["tone_n"] else np.nan,
            "goldstein_latest": record["gold_sum"]/n if n else np.nan,
            "goldstein_std_latest": np.sqrt(variance),
            "total_events_latest": float(total),
            "domains_latest": float(len(record["domains"])),
            "conflict_share_latest": panel.iloc[index, index]/total if total else np.nan,
            "news_volume_change_wow": total/previous-1 if previous and previous_week not in bad_weeks else np.nan,
            **{f"root_{code}_latest": float(record["roots"].get(code, 0)) for code in ROOT_CODES},
        }
        if week in bad_weeks:
            for key in news.keys() - {"week_start", "forecast_issued_at_utc", "asof_export_before"}:
                news[key] = np.nan
        records.append(news)
    truth_7d = grace_truth(panel, missing, legacy_grace_before)
    count_weeks = weeks < pd.Timestamp(truth_cutoff)-pd.Timedelta(days=7)
    truth_2024 = pd.Series(final_counts[count_weeks], index=weeks[count_weeks], name="conflict_count")
    truth_2024.loc[truth_2024.index.isin(bad_weeks)] = np.nan
    audit["duplicate_rows"] = audit["raw_rows"]-audit["unique_rows"]
    audit["downloaded_at"] = [min(download_dates), max(download_dates)] if download_dates else []
    return {
        "count_vintages.parquet": panel,
        "news_asof.csv": pd.DataFrame(records).set_index("week_start"),
        "truth_7d.csv": truth_7d.to_frame(),
        "truth_2024.csv": truth_2024.to_frame(),
    }, audit


def prepare_ecb(raw):
    directory = Path(raw)/"external"
    metadata = json.loads((directory/"ecb_ils.meta.json").read_text())
    archive = directory/"ecb_eurofxref_hist.zip"
    if sha256(archive) != metadata["archive_sha256"]:
        raise ValueError("ECB archive checksum mismatch")
    with zipfile.ZipFile(archive) as zipped:
        frame = pd.read_csv(zipped.open("eurofxref-hist.csv"), usecols=["Date", "ILS", "USD"])
    frame["Date"] = pd.to_datetime(frame.Date)
    frame = frame.dropna().sort_values("Date")
    frame["ILS_per_USD"] = frame.ILS/frame.USD
    return frame.set_index("Date"), metadata


def verify_against(output, reference):
    manifest = json.loads((Path(output)/"manifest.json").read_text())
    for name in manifest["files"]:
        left, right = Path(output)/name, Path(reference)/name
        if name.endswith(".parquet"):
            actual, expected = pd.read_parquet(left), pd.read_parquet(right)
        else:
            actual = pd.read_csv(left, index_col=0, parse_dates=[0])
            expected = pd.read_csv(right, index_col=0, parse_dates=[0])
            for column in ("forecast_issued_at_utc", "asof_export_before"):
                if column in actual:
                    actual[column] = pd.to_datetime(actual[column], format="mixed")
                    expected[column] = pd.to_datetime(expected[column], format="mixed")
        pd.testing.assert_frame_equal(actual, expected, check_freq=False, check_dtype=False,
                                      check_exact=False, rtol=1e-11, atol=1e-11)
    print("Verified: all five snapshots match the reference (values, dates and missingness).")


def build(raw, output, strict_grace=False):
    frames, audit = aggregate(
        daily_files(raw), pd.date_range(FIRST_WEEK, LAST_WEEK, freq="W-MON"),
        legacy_grace_before=None if strict_grace else "2024-12-30",
    )
    frames["ecb_ils.csv"], ecb_meta = prepare_ecb(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if name.endswith(".parquet"):
            frame.to_parquet(output/name)
        else:
            frame.to_csv(output/name)
    ecb_meta = {**ecb_meta, "sha256": sha256(output/"ecb_ils.csv")}
    manifest = {
        "country": "Israel (GDELT FIPS IS; ISO ISR)",
        "gdelt_version": "1.0 daily Events, 58 columns",
        "gdelt_url": "https://data.gdeltproject.org/events/{YYYYMMDD}.export.CSV.zip",
        "gdelt_index": "https://data.gdeltproject.org/events/",
        "archive_range": [FIRST_EXPORT, LAST_EXPORT],
        "downloaded_at": audit["downloaded_at"],
        "weekly_calendar": "Monday-Sunday; SQLDATE event week; unique GLOBALEVENTID across exports; ActionGeo_CountryCode=IS; roots 18/19",
        "snapshots": "Rows are origins, columns are event weeks; only columns <= origin may be used; export_day < origin Monday +7d.",
        "news": "Own-week records available by issue; news_volume_change_wow denominator has one-week grace.",
        "issue": "Following Monday 12:00 UTC, assumed publisher schedule; no started-week archives.",
        "truth_2024": "All fetched archives through 2025-01-05; CV early truth panel row166, late truth row195.",
        "truth_7d": (
            "Each target week with 7-day grace; own or grace week missing -> NaN."
            if strict_grace else
            "Reproduce the original benchmark: before 2024-12-30 only own-week gaps are masked; "
            "from 2024-12-30 own or grace-week gaps -> NaN. "
            "Legacy labels 2022-10-31 and 2023-03-13 have incomplete grace; use --strict-grace to mask them."
        ),
        "missing_exports": MISSING,
        "ecb": ecb_meta,
        "files": {name: sha256(output/name) for name in frames},
        "preprocessing": {"script_sha256": sha256(Path(__file__)),
                          "strict_grace": strict_grace, **audit},
    }
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT/"data/raw")
    parser.add_argument("--output", type=Path, default=ROOT/"data/preprocessed_rebuilt")
    parser.add_argument("--verify-against", type=Path, help="Optional reference; never used to build data")
    parser.add_argument("--strict-grace", action="store_true",
                        help="Also mask two incomplete pre-2025 grace labels; changes the historical benchmark")
    args = parser.parse_args()
    if not args.strict_grace:
        print("Reproducing legacy labels. --strict-grace additionally masks 2022-10-31 and 2023-03-13.", flush=True)
    build(args.raw, args.output, args.strict_grace)
    if args.verify_against:
        verify_against(args.output, args.verify_against)
    print(f"Prepared snapshots: {args.output}")


if __name__ == "__main__":
    main()
