import pyarrow as pa
import pyarrow.parquet as pq
from math import ceil
from pathlib import Path

def write_parquet_shards(rows: list[dict], out_dir: str, prefix: str, shard_size: int = 500_000, compression: str = "zstd"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = len(rows)
    num_shards = ceil(n / shard_size)
    for i in range(num_shards):
        shard = rows[i*shard_size : (i+1)*shard_size]
        table = pa.Table.from_pylist(shard)  # infers schema: text=str, label=int, a=int, b=int, op=str
        pq.write_table(
            table,
            f"{out_dir}/{prefix}-{i:05d}.parquet",
            compression=compression,
            use_dictionary=["text", "op"],  # improves compression
        )
