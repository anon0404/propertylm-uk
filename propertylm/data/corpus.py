from __future__ import annotations
from pathlib import Path
import duckdb

EPC_ALIASES = {
    "LMK_KEY": "epc_id", "UPRN": "uprn", "ADDRESS": "raw_address",
    "ADDRESS1": "address1", "ADDRESS2": "address2", "ADDRESS3": "address3",
    "POSTCODE": "postcode", "POSTTOWN": "town", "INSPECTION_DATE": "inspection_date",
}


def _column_expr(columns: set[str], candidates: list[str], alias: str) -> str:
    for c in candidates:
        if c in columns:
            return f'CAST("{c}" AS VARCHAR) AS {alias}'
    return f"'' AS {alias}"


def build_corpus(epc_dir: Path, uprn_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    epc_glob = str(epc_dir / "*.csv")
    sample = con.execute(f"SELECT * FROM read_csv_auto('{epc_glob}', union_by_name=true, sample_size=20000) LIMIT 0")
    cols = {d[0] for d in sample.description}
    select = [
        _column_expr(cols, ["LMK_KEY", "lmk-key", "lmk_key"], "epc_id"),
        _column_expr(cols, ["UPRN", "uprn"], "uprn"),
        _column_expr(cols, ["ADDRESS", "address"], "raw_address"),
        _column_expr(cols, ["ADDRESS1", "address1"], "address1"),
        _column_expr(cols, ["ADDRESS2", "address2"], "address2"),
        _column_expr(cols, ["ADDRESS3", "address3"], "address3"),
        _column_expr(cols, ["POSTCODE", "postcode"], "postcode"),
        _column_expr(cols, ["POSTTOWN", "posttown", "town"], "town"),
        _column_expr(cols, ["INSPECTION_DATE", "inspection-date", "inspection_date"], "inspection_date"),
    ]
    uprn_files = list(uprn_dir.glob("*.parquet")) + list(uprn_dir.glob("*.csv"))
    if not uprn_files:
        uprn_join = "SELECT '' uprn, NULL::DOUBLE x, NULL::DOUBLE y WHERE false"
    else:
        pattern = str(uprn_dir / ("*.parquet" if list(uprn_dir.glob('*.parquet')) else "*.csv"))
        reader = f"read_parquet('{pattern}', union_by_name=true)" if pattern.endswith("parquet") else f"read_csv_auto('{pattern}', union_by_name=true)"
        uprn_join = f"SELECT CAST(UPRN AS VARCHAR) uprn, TRY_CAST(X_COORDINATE AS DOUBLE) x, TRY_CAST(Y_COORDINATE AS DOUBLE) y FROM {reader}"
    sql = f"""
    COPY (
      WITH epc AS (
        SELECT {', '.join(select)}
        FROM read_csv_auto('{epc_glob}', union_by_name=true, sample_size=20000, ignore_errors=true)
      ), clean AS (
        SELECT *,
          upper(regexp_replace(trim(postcode), '\\s+', '', 'g')) AS pc_compact,
          trim(concat_ws(', ', nullif(raw_address,''), nullif(address1,''), nullif(address2,''), nullif(address3,''), nullif(town,''))) AS full_address,
          row_number() over(partition by coalesce(nullif(uprn,''), md5(coalesce(raw_address,'') || postcode)) order by inspection_date desc nulls last) rn
        FROM epc
      ), u AS ({uprn_join})
      SELECT
        'epc_' || row_number() over() AS property_id,
        epc_id, c.uprn, full_address AS raw_address, town,
        CASE WHEN length(pc_compact)>=5 THEN substr(pc_compact,1,length(pc_compact)-3)||' '||right(pc_compact,3) ELSE pc_compact END postcode,
        inspection_date, u.x, u.y,
        full_address || ' | postcode=' || CASE WHEN length(pc_compact)>=5 THEN substr(pc_compact,1,length(pc_compact)-3)||' '||right(pc_compact,3) ELSE pc_compact END AS canonical_text
      FROM clean c LEFT JOIN u USING(uprn)
      WHERE rn=1 AND full_address<>''
    ) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(sql)
    count = con.execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0]
    print(f"Wrote {count:,} deduplicated properties to {output}")
