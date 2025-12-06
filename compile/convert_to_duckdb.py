"""
Convert 22,464 CSV files into a single DuckDB database.
This eliminates I/O overhead and prevents memory crashes.
"""

import duckdb
import os
import glob
import re
from tqdm import tqdm
from pathlib import Path


def parse_config_name(config_name: str):
    """Extract structured info from config folder name"""
    segments = config_name.split("__")
    config = {}

    for seg in segments:
        if "_" in seg:
            key, value = seg.split("_", 1)
            config[key] = value
        else:
            config[seg] = True

    for k, v in config.items():
        if isinstance(v, str) and re.match(r"^-?\d+(\.\d+)?$", v):
            config[k] = float(v)

    return config


def convert_csvs_to_duckdb(base_dir, db_path="sumobot_data.duckdb"):
    """
    Read all CSV files and store them in a single DuckDB database.

    This solves the "22,464 files" problem by:
    1. Consolidating all data into one database file
    2. Creating indexes for fast queries
    3. No memory loading - DuckDB handles everything on disk

    Args:
        base_dir: Root directory containing matchup folders
        db_path: Output DuckDB database file (will be ~20-40GB compressed)
    """

    # Connect to DuckDB (creates file if doesn't exist)
    con = duckdb.connect(db_path)

    print(f"🦆 Creating DuckDB database: {db_path}")

    # Find all matchup folders
    matchup_folders = [f for f in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, f))]

    print(f"📂 Found {len(matchup_folders)} matchup folders")

    # Find first CSV to detect schema
    print(f"🔍 Detecting CSV schema from first file...")
    first_csv = None
    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]
        for config_folder in config_folders:
            config_path = os.path.join(matchup_path, config_folder)
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))
            if csv_files:
                first_csv = csv_files[0]
                break
        if first_csv:
            break

    if not first_csv:
        print("❌ No CSV files found!")
        con.close()
        return

    # Create table with ALL columns from CSV + metadata columns
    print(f"📋 Using schema from: {first_csv}")
    con.execute(f"""
        CREATE TABLE game_logs AS
        SELECT
            *,
            '' as bot_a,
            '' as bot_b,
            '' as config_name,
            '' as csv_source,
            '' as SkillLeft,
            '' as SkillRight
        FROM read_csv_auto('{first_csv}')
        WHERE 1=0  -- Don't insert data yet, just create schema
    """)

    print(f"✅ Table created with all CSV columns + SkillLeft/SkillRight")

    print(f"📝 Processing CSV files...\n")

    total_inserted = 0

    for matchup_folder in tqdm(matchup_folders, desc="Converting matchups"):
        matchup_path = os.path.join(base_dir, matchup_folder)

        # Extract bot names
        match = re.match(r"(.+)_vs_(.+)", matchup_folder)
        if not match:
            print(f"⚠️  Skipping invalid matchup name: {matchup_folder}")
            continue

        bot_a, bot_b = match.groups()

        # Find all config folders
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        for config_folder in config_folders:
            config_path = os.path.join(matchup_path, config_folder)
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))

            # Parse config to extract skills
            config = parse_config_name(config_folder)
            skill_left = config.get('SkillLeft', '')
            skill_right = config.get('SkillRight', '')

            for csv_file in csv_files:
                try:
                    # DuckDB reads CSV directly without loading into memory
                    con.execute(f"""
                        INSERT INTO game_logs
                        SELECT
                            *,
                            '{bot_a}' as bot_a,
                            '{bot_b}' as bot_b,
                            '{config_folder}' as config_name,
                            '{csv_file}' as csv_source,
                            '{skill_left}' as SkillLeft,
                            '{skill_right}' as SkillRight
                        FROM read_csv_auto('{csv_file}')
                    """)
                    total_inserted += 1

                except Exception as e:
                    print(f"\n❌ Error reading {csv_file}: {e}")
                    with open("conversion_errors.log", "a") as f:
                        f.write(f"{csv_file}: {e}\n")

    print(f"\n✅ Successfully inserted {total_inserted} CSV files into database")

    # Create indexes for fast queries
    print("🔍 Creating indexes and constraints...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_game ON game_logs(GameIndex)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_bot_a ON game_logs(bot_a)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_bot_b ON game_logs(bot_b)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_config ON game_logs(config_name)")

    # Create unique index on the combination to prevent duplicate entries
    # Note: This creates a unique constraint on game identity
    print("🔒 Creating unique constraint on (GameIndex, bot_a, bot_b, config_name)...")
    try:
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_game
            ON game_logs(GameIndex, bot_a, bot_b, config_name)
        """)
        print("✅ Unique constraint created")
    except Exception as e:
        print(f"⚠️  Warning: Could not create unique constraint: {e}")
        print("   (This might mean you have duplicate data)")

    # Show stats
    stats = con.execute("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT GameIndex) as total_games,
            COUNT(DISTINCT bot_a || '_vs_' || bot_b) as total_matchups,
            COUNT(DISTINCT config_name) as total_configs
        FROM game_logs
    """).fetchone()

    print(f"\n📊 Database Statistics:")
    print(f"   Total rows: {stats[0]:,}")
    print(f"   Total games: {stats[1]:,}")
    print(f"   Total matchups: {stats[2]:,}")
    print(f"   Total configs: {stats[3]:,}")

    # Show database size
    db_size = os.path.getsize(db_path) / (1024**3)  # GB
    print(f"   Database size: {db_size:.2f} GB")

    con.close()
    print(f"\n🎉 Conversion complete! Database saved to: {db_path}")


def verify_database(db_path="sumobot_data.duckdb"):
    """Quick verification that database is working"""
    con = duckdb.connect(db_path, read_only=True)

    print("🔍 Verifying database...")

    # Sample query
    sample = con.execute("""
        SELECT bot_a, bot_b, config_name, COUNT(*) as row_count
        FROM game_logs
        GROUP BY bot_a, bot_b, config_name
        LIMIT 5
    """).fetchdf()

    print("\n📋 Sample data:")
    print(sample)

    con.close()
    print("✅ Database is working correctly!")


if __name__ == "__main__":
    import sys

    BASE_DIR = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"

    # if len(sys.argv) < 2:
    #     print("Usage: python convert_to_duckdb.py <base_dir> [output_db_path]")
    #     print("Example: python convert_to_duckdb.py ./data sumobot.duckdb")
    #     sys.exit(1)

    base_dir = BASE_DIR
    db_path = "sumobot_data.duckdb"

    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        sys.exit(1)

    # Convert
    convert_csvs_to_duckdb(base_dir, db_path)

    # Verify
    verify_database(db_path)
