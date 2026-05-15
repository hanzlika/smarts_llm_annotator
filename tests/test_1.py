import pandas as pd

from pathlib import Path
from scripts import smarts_annotation_pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]

csv_path = ROOT_DIR / 'data' / 'test_data.csv'
out_path = ROOT_DIR / 'data' / 'test_output.csv'

# if this works out, it will generate an output csv
smarts_annotation_pipeline.run(csv_path, smarts_col='smarts', out_path=out_path ,use_local=True, ignore_time_window=True)

df = pd.read_csv(out_path)

print("Test results:")
print(df)

