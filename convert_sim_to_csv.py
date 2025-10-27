import os
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Optional


def extract_game_index(filename: str) -> int:
    """Extract game index from filename like 'game_001' -> 1"""
    match = re.match(r"game_(\d+)", filename)
    return int(match.group(1)) if match else -1


def escape_csv(value: str) -> str:
    """Escape CSV value if it contains special characters"""
    if ',' in value or '"' in value or '\n' in value:
        return f'"{value.replace(chr(34), chr(34) + chr(34))}"'
    return value


def convert_logs_to_csv(folder_path: str, output_path: str, skip_if_exists: bool = True):
    """Convert JSON game logs to CSV format"""
    try:
        # Check if any CSV file already exists in the folder
        if skip_if_exists:
            existing_csvs = list(Path(folder_path).glob("*.csv"))
            if existing_csvs:
                print(f"⊘ Skipping (CSV already exists): {Path(folder_path).name}")
                return
        
        csv_rows = []
        
        # Find and sort game JSON files
        game_files = [
            f for f in Path(folder_path).glob("game_*.json")
            if re.match(r"game_\d+\.json", f.name)
        ]
        game_files.sort(key=lambda f: extract_game_index(f.stem))
        
        for file in game_files:
            with open(file, 'r') as f:
                root = json.load(f)
            
            game_index = root.get("Index", -1)
            game_timestamp = root.get("Timestamp", "")
            game_winner = root.get("Winner", "")
            
            for round_data in root.get("Rounds", []):
                round_index = round_data.get("Index", -1)
                round_timestamp = round_data.get("Timestamp", "")
                round_winner = round_data.get("Winner", "")
                
                for event_log in round_data.get("PlayerEvents", []):
                    if event_log.get("Category") == "LastPosition":
                        continue
                    
                    # Map winner values
                    game_winner_val = "2" if game_winner == "Draw" else "0" if game_winner == "Left" else "1"
                    round_winner_val = "2" if round_winner == "Draw" else "0" if round_winner == "Left" else "1"
                    
                    row = {
                        "GameIndex": str(game_index + 1),
                        "GameWinner": game_winner_val,
                        "GameTimestamp": game_timestamp,
                        "RoundIndex": str(round_index),
                        "RoundWinner": round_winner_val,
                        "RoundTimestamp": round_timestamp,
                        "StartedAt": event_log.get("StartedAt", ""),
                        "UpdatedAt": event_log.get("UpdatedAt", ""),
                        "Actor": "0" if event_log.get("Actor") == "Left" else "1",
                        "Target": "" if event_log.get("Target", "") == "" else "0" if event_log.get("Target") == "Left" else "1",
                        "Category": event_log.get("Category", ""),
                        "State": str(event_log.get("State", "")),
                    }
                    
                    # Extract action data
                    act = event_log.get("Data", {})
                    if act:
                        row["Name"] = act.get("Name", "")
                        row["Duration"] = str(act.get("Duration", ""))
                        row["Reason"] = act.get("Reason", "")
                        
                        # Robot data
                        robot = act.get("Robot", {})
                        if robot:
                            pos = robot.get("Position", {})
                            row["BotPosX"] = str(pos.get("X", ""))
                            row["BotPosY"] = str(pos.get("Y", ""))
                            row["BotLinv"] = str(robot.get("LinearVelocity", ""))
                            row["BotAngv"] = str(robot.get("AngularVelocity", ""))
                            row["BotRot"] = str(robot.get("Rotation", ""))
                            row["BotIsDashActive"] = "1" if robot.get("IsDashActive") else "0"
                            row["BotIsSkillActive"] = "1" if robot.get("IsSkillActive") else "0"
                            row["BotIsOutFromArena"] = "1" if robot.get("IsOutFromArena") else "0"
                        
                        # Enemy robot data
                        enemy_robot = act.get("EnemyRobot", {})
                        if enemy_robot:
                            enemy_pos = enemy_robot.get("Position", {})
                            row["EnemyBotPosX"] = str(enemy_pos.get("X", ""))
                            row["EnemyBotPosY"] = str(enemy_pos.get("Y", ""))
                            row["EnemyBotLinv"] = str(enemy_robot.get("LinearVelocity", ""))
                            row["EnemyBotAngv"] = str(enemy_robot.get("AngularVelocity", ""))
                            row["EnemyBotRot"] = str(enemy_robot.get("Rotation", ""))
                            row["EnemyBotIsDashActive"] = "1" if enemy_robot.get("IsDashActive") else "0"
                            row["EnemyBotIsSkillActive"] = "1" if enemy_robot.get("IsSkillActive") else "0"
                            row["EnemyBotIsOutFromArena"] = "1" if enemy_robot.get("IsOutFromArena") else "0"
                    
                    # Collision data
                    if event_log.get("Category") == "Collision":
                        collision_data = event_log.get("Data", {})
                        row["ColActor"] = str(collision_data.get("IsActor", ""))
                        row["ColImpact"] = str(collision_data.get("Impact", ""))
                        row["ColTieBreaker"] = str(collision_data.get("IsTieBreaker", ""))
                        row["ColLockDuration"] = str(collision_data.get("LockDuration", ""))
                        
                        col_robot = collision_data.get("Robot", {})
                        if col_robot:
                            col_pos = col_robot.get("Position", {})
                            row["ColBotPosX"] = str(col_pos.get("X", ""))
                            row["ColBotPosY"] = str(col_pos.get("Y", ""))
                            row["ColBotLinv"] = str(col_robot.get("LinearVelocity", ""))
                            row["ColBotAngv"] = str(col_robot.get("AngularVelocity", ""))
                            row["ColBotRot"] = str(col_robot.get("Rotation", ""))
                            row["ColBotIsDashActive"] = "1" if col_robot.get("IsDashActive") else "0"
                            row["ColBotIsSkillActive"] = "1" if col_robot.get("IsSkillActive") else "0"
                            row["ColBotIsOutFromArena"] = "1" if col_robot.get("IsOutFromArena") else "0"
                        
                        col_enemy = collision_data.get("EnemyRobot", {})
                        if col_enemy:
                            col_enemy_pos = col_enemy.get("Position", {})
                            row["ColEnemyBotPosX"] = str(col_enemy_pos.get("X", ""))
                            row["ColEnemyBotPosY"] = str(col_enemy_pos.get("Y", ""))
                            row["ColEnemyBotLinv"] = str(col_enemy.get("LinearVelocity", ""))
                            row["ColEnemyBotAngv"] = str(col_enemy.get("AngularVelocity", ""))
                            row["ColEnemyBotRot"] = str(col_enemy.get("Rotation", ""))
                            row["ColEnemyBotIsDashActive"] = "1" if col_enemy.get("IsDashActive") else "0"
                            row["ColEnemyBotIsSkillActive"] = "1" if col_enemy.get("IsSkillActive") else "0"
                            row["ColEnemyBotIsOutFromArena"] = "1" if col_enemy.get("IsOutFromArena") else "0"
                    
                    csv_rows.append(row)
        
        # Write CSV
        if csv_rows:
            # Define column order
            column_order = [
                "GameIndex", "GameWinner", "GameTimestamp",
                "RoundIndex", "RoundWinner", "RoundTimestamp",
                "StartedAt", "UpdatedAt", "Actor", "Target",
                "Category", "State", "Name", "Duration", "Reason",
                "BotPosX", "BotPosY", "BotLinv", "BotAngv", "BotRot",
                "BotIsDashActive", "BotIsSkillActive", "BotIsOutFromArena",
                "EnemyBotPosX", "EnemyBotPosY", "EnemyBotLinv", "EnemyBotAngv", "EnemyBotRot",
                "EnemyBotIsDashActive", "EnemyBotIsSkillActive", "EnemyBotIsOutFromArena",
                "ColActor", "ColImpact", "ColTieBreaker", "ColLockDuration",
                "ColBotPosX", "ColBotPosY", "ColBotLinv", "ColBotAngv", "ColBotRot",
                "ColBotIsDashActive", "ColBotIsSkillActive", "ColBotIsOutFromArena",
                "ColEnemyBotPosX", "ColEnemyBotPosY", "ColEnemyBotLinv", "ColEnemyBotAngv", "ColEnemyBotRot",
                "ColEnemyBotIsDashActive", "ColEnemyBotIsSkillActive", "ColEnemyBotIsOutFromArena"
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=column_order, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"✓ Generated: {output_path}")
        else:
            print(f"⚠ No data found in: {folder_path}")
    
    except Exception as ex:
        print(f"✗ Error processing {folder_path}: {ex}")
        raise


def convert_all_configs(base_dir: str, included_agents: str = "*"):
    """
    Convert all simulation logs to CSV
    
    Args:
        base_dir: Base directory containing simulation data
        included_agents: Filter agents (use "*" for all, or separate by ";")
    """
    try:
        # Parse included agents
        included_list = None if included_agents == "*" else [
            agent.strip() for agent in included_agents.split(";") if agent.strip()
        ]
        
        # Find all Timer_* folders
        config_folders = []
        for root, dirs, files in os.walk(base_dir):
            for dir_name in dirs:
                if dir_name.startswith("Timer_"):
                    config_folders.append(os.path.join(root, dir_name))
        
        # Filter by included agents
        if included_list:
            config_folders = [
                folder for folder in config_folders
                if any(agent.lower() in Path(folder).parent.name.lower() 
                      for agent in included_list)
            ]
        
        if not config_folders:
            print("No matching folders found to convert.")
            return
        
        print(f"Found {len(config_folders)} folder(s) to process\n")
        
        # Process each folder
        for i, config_folder in enumerate(config_folders, 1):
            config_name = Path(config_folder).name
            output_path = os.path.join(config_folder, f"{config_name}.csv")
            
            # Skip if CSV already exists
            if os.path.exists(output_path):
                print(f"[{i}/{len(config_folders)}] Skipping {config_name} (CSV exists)")
                continue
            
            print(f"[{i}/{len(config_folders)}] Processing {config_name}...")
            convert_logs_to_csv(config_folder, output_path)
        
        print(f"\n✓ All CSV files generated successfully!")
    
    except Exception as ex:
        print(f"\n✗ Error: {ex}")
        raise


def test_single_folder(folder_path: str):
    """
    Test conversion on a single Timer_* folder
    
    Args:
        folder_path: Path to a single Timer_* folder to convert
    """
    try:
        folder_name = Path(folder_path).name
        
        if not os.path.exists(folder_path):
            print(f"✗ Folder not found: {folder_path}")
            return
        
        if not folder_name.startswith("Timer_"):
            print(f"⚠ Warning: Folder '{folder_name}' doesn't start with 'Timer_'")
        
        output_path = os.path.join(folder_path, f"{folder_name}.csv")
        
        print(f"Testing conversion on: {folder_name}")
        print(f"Output: {output_path}\n")
        
        convert_logs_to_csv(folder_path, output_path)
        
        print(f"\n✓ Test completed successfully!")
        
    except Exception as ex:
        print(f"\n✗ Test failed: {ex}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert simulation JSON logs to CSV")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert all command
    convert_parser = subparsers.add_parser("convert", help="Convert all matching folders")
    convert_parser.add_argument("base_dir", help="Base directory containing simulation data")
    convert_parser.add_argument(
        "--agents", 
        default="*", 
        help="Filter agents (use '*' for all, or separate by ';'). Example: 'Bot_A;Bot_B'"
    )
    
    # Test single folder command
    test_parser = subparsers.add_parser("test", help="Test conversion on a single folder")
    test_parser.add_argument("folder_path", help="Path to a single Timer_* folder")
    
    args = parser.parse_args()
    
    if args.command == "convert":
        convert_all_configs(args.base_dir, args.agents)
    elif args.command == "test":
        test_single_folder(args.folder_path)
    else:
        parser.print_help()