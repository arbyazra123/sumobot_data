import sys
import traceback
import pandas as pd
from pathlib import Path
from generator import process_log

def main():
    if len(sys.argv) < 4:
        print("Usage: process_one.py <csv_path> <bot_a> <bot_b> <config_name>")
        sys.exit(1)

    csv_path, bot_a, bot_b, config_name, chunksize = sys.argv[1:6]
    csv_path = Path(csv_path)

    try:
        df, df_time = process_log(csv_path, bot_a, bot_b, config_name, int(chunksize))
        out_path = csv_path.with_suffix(".processed.csv")
        out_time_path = csv_path.with_suffix(".time.processed.csv")
        df.to_csv(out_path, index=False)
        df_time.to_csv(out_time_path, index=False)
        print(f"[OK] {csv_path.name} -> {out_path.name}")
    except Exception as e:
        print(f"[ERROR] {csv_path.name}: {e}")
        traceback.print_exc(-1)
        with open("failed_files.txt", "a") as f:
            f.write(f"{csv_path} err: {e}\n")

if __name__ == "__main__":
    main()
