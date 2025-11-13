#%%
from data_generator import DataGenerator
from dataset_utils import write_parquet_shards, peek_parquet

if __name__ == '__main__':
    gen = DataGenerator(seed=42)
    data1 = gen.synthetic_data_one(1000)
    write_parquet_shards(data1, "output", "data-18-digit")
    peek_parquet("output/data-18-digit-00000.parquet")

# %%
