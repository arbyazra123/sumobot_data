import os
import re
import json
import csv
from glob import glob
from tqdm import tqdm  # optional: pip install tqdm


def extract_game_index(filename: str) -> int:
    """Extract numeric index from filename like 'game_001.json'."""
    match = re.search(r"game_(\d+)", filename)
    return int(match.group(1)) if match else -1


def escape_csv(value: str) -> str:
    """Escape CSV fields like C# version."""
    if any(c in value for c in [',', '"', '\n']):
        return '"' + value.replace('"', '""') + '"'
    return value


def convert_logs_to_csv(folder_path: str, output_path: str):
    """Convert all game_*.json files in folder to one CSV."""
    csv_rows = []

    files = sorted(
        glob(os.path.join(folder_path, "game_*.json")),
        key=lambda f: extract_game_index(os.path.basename(f))
    )

    for file in tqdm(files, desc=f"Processing {folder_path}", ncols=100):
        with open(file, "r", encoding="utf-8") as f:
            root = json.load(f)

        game_index = root.get("Index", -1)
        game_timestamp = root.get("Timestamp", "")
        game_winner = root.get("Winner", "")

        rounds = root.get("Rounds", [])
        for round_data in rounds:
            round_index = round_data.get("Index", -1)
            round_timestamp = round_data.get("Timestamp", "")
            round_winner = round_data.get("Winner", "")

            player_events = round_data.get("PlayerEvents", [])
            for event_log in player_events:
                if event_log.get("Category") == "LastPosition":
                    continue

                row = {
                    "GameIndex": str(game_index + 1),
                    "GameWinner": "2" if game_winner == "Draw" else "0" if game_winner == "Left" else "1",
                    "GameTimestamp": game_timestamp,
                    "RoundIndex": str(round_index),
                    "RoundWinner": "2" if round_winner == "Draw" else "0" if round_winner == "Left" else "1",
                    "RoundTimestamp": round_timestamp,
                    "StartedAt": str(event_log.get("StartedAt", "")),
                    "UpdatedAt": str(event_log.get("UpdatedAt", "")),
                    "Actor": "0" if event_log.get("Actor") == "Left" else "1",
                }

                target = event_log.get("Target", "")
                row["Target"] = "" if target == "" else "0" if target == "Left" else "1"
                row["Category"] = str(event_log.get("Category", ""))
                row["State"] = str(event_log.get("State", ""))

                act = event_log.get("Data")
                if act:
                    row["Name"] = str(act.get("Name", ""))
                    row["Duration"] = str(act.get("Duration", ""))
                    row["Reason"] = "" if str(act.get("Reason", "")) == "None" else str(act.get("Reason", ""))

                    robot = act.get("Robot")
                    if robot:
                        pos = robot.get("Position", {})
                        row.update({
                            "BotPosX": str(pos.get("X", "")),
                            "BotPosY": str(pos.get("Y", "")),
                            "BotLinv": str(robot.get("LinearVelocity", "")),
                            "BotAngv": str(robot.get("AngularVelocity", "")),
                            "BotRot": str(robot.get("Rotation", "")),
                            "BotIsDashActive": "1" if robot.get("IsDashActive") else "0",
                            "BotIsSkillActive": "1" if robot.get("IsSkillActive") else "0",
                            "BotIsOutFromArena": "1" if robot.get("IsOutFromArena") else "0",
                        })

                    enemy = act.get("EnemyRobot")
                    if enemy:
                        pos = enemy.get("Position", {})
                        row.update({
                            "EnemyBotPosX": str(pos.get("X", "")),
                            "EnemyBotPosY": str(pos.get("Y", "")),
                            "EnemyBotLinv": str(enemy.get("LinearVelocity", "")),
                            "EnemyBotAngv": str(enemy.get("AngularVelocity", "")),
                            "EnemyBotRot": str(enemy.get("Rotation", "")),
                            "EnemyBotIsDashActive": "1" if enemy.get("IsDashActive") else "0",
                            "EnemyBotIsSkillActive": "1" if enemy.get("IsSkillActive") else "0",
                            "EnemyBotIsOutFromArena": "1" if enemy.get("IsOutFromArena") else "0",
                        })

                if event_log.get("Category") == "Collision":
                    col_data = event_log.get("Data", {})
                    row["ColActor"] = str(col_data.get("IsActor", ""))
                    row["ColImpact"] = str(col_data.get("Impact", ""))
                    row["ColTieBreaker"] = str(col_data.get("IsTieBreaker", ""))
                    row["ColLockDuration"] = str(col_data.get("LockDuration", ""))

                    col_robot = col_data.get("Robot")
                    if col_robot:
                        pos = col_robot.get("Position", {})
                        row.update({
                            "ColBotPosX": str(pos.get("X", "")),
                            "ColBotPosY": str(pos.get("Y", "")),
                            "ColBotLinv": str(col_robot.get("LinearVelocity", "")),
                            "ColBotAngv": str(col_robot.get("AngularVelocity", "")),
                            "ColBotRot": str(col_robot.get("Rotation", "")),
                            "ColBotIsDashActive": "1" if col_robot.get("IsDashActive") else "0",
                            "ColBotIsSkillActive": "1" if col_robot.get("IsSkillActive") else "0",
                            "ColBotIsOutFromArena": "1" if col_robot.get("IsOutFromArena") else "0",
                        })

                    col_enemy = col_data.get("EnemyRobot")
                    if col_enemy:
                        pos = col_enemy.get("Position", {})
                        row.update({
                            "ColEnemyBotPosX": str(pos.get("X", "")),
                            "ColEnemyBotPosY": str(pos.get("Y", "")),
                            "ColEnemyBotLinv": str(col_enemy.get("LinearVelocity", "")),
                            "ColEnemyBotAngv": str(col_enemy.get("AngularVelocity", "")),
                            "ColEnemyBotRot": str(col_enemy.get("Rotation", "")),
                            "ColEnemyBotIsDashActive": "1" if col_enemy.get("IsDashActive") else "0",
                            "ColEnemyBotIsSkillActive": "1" if col_enemy.get("IsSkillActive") else "0",
                            "ColEnemyBotIsOutFromArena": "1" if col_enemy.get("IsOutFromArena") else "0",
                        })

                csv_rows.append(row)

    # Collect all CSV columns
    preferred_order = [
        "GameIndex","GameWinner","GameTimestamp","RoundIndex","RoundWinner","RoundTimestamp","StartedAt","UpdatedAt","Actor","Target","Category","State","Name","Duration","Reason","BotPosX","BotPosY","BotLinv","BotAngv","BotRot","BotIsDashActive","BotIsSkillActive","BotIsOutFromArena","EnemyBotPosX","EnemyBotPosY","EnemyBotLinv","EnemyBotAngv","EnemyBotRot","EnemyBotIsDashActive","EnemyBotIsSkillActive","EnemyBotIsOutFromArena","ColActor","ColImpact","ColTieBreaker","ColLockDuration","ColBotPosX","ColBotPosY","ColBotLinv","ColBotAngv","ColBotRot","ColBotIsDashActive","ColBotIsSkillActive","ColBotIsOutFromArena","ColEnemyBotPosX","ColEnemyBotPosY","ColEnemyBotLinv","ColEnemyBotAngv","ColEnemyBotRot","ColEnemyBotIsDashActive","ColEnemyBotIsSkillActive","ColEnemyBotIsOutFromArena"
    ]
    # Merge preferred order with dynamically discovered keys
    all_keys = preferred_order + [k for k in {kk for d in csv_rows for kk in d.keys()} if k not in preferred_order]


    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(all_keys)
        for row in csv_rows:
            writer.writerow([row.get(k, "") for k in all_keys])

    print(f"✅ Saved CSV: {output_path}")


def convert_all_configs(simulation_root: str):
    """Convert all config folders recursively (Timer_*)."""
    config_folders = []
    for root, dirs, _ in os.walk(simulation_root):
        for d in dirs:
            if d.startswith("Timer_"):
                config_folders.append(os.path.join(root, d))

    for i, config_folder in enumerate(config_folders, 1):
        config_name = os.path.basename(config_folder)
        output_path = os.path.join(config_folder, f"{config_name}.csv")

        print(f"[{i}/{len(config_folders)}] Processing {config_name}")
        convert_logs_to_csv(config_folder, output_path)


if __name__ == "__main__":
    # Example usage:
    incompletes = [
        ["Bot_GA_vs_Bot_Primitive","Timer_60__ActInterval_0.1__Round_BestOf5__SkillLeft_Boost__SkillRight_Boost"],
        ["Bot_FSM_vs_Bot_Primitive","Timer_15__ActInterval_0.1__Round_BestOf3__SkillLeft_Boost__SkillRight_Boost"],
        ["Bot_BT_vs_Bot_UtilityAI","Timer_45__ActInterval_0.1__Round_BestOf5__SkillLeft_Stone__SkillRight_Stone"],
        ["Bot_UtilityAI_vs_Bot_FSM","Timer_15__ActInterval_0.1__Round_BestOf5__SkillLeft_Boost__SkillRight_Stone"]
    ]
    simulation_root = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"

    for inc in incompletes:
        name = inc[1]
        specific_folder = os.path.join(
            simulation_root,
            inc[0],
            name,
        )
        convert_logs_to_csv(specific_folder, os.path.join(specific_folder, f"{name}.csv"))


    # OR convert all configs:
    # convert_all_configs(simulation_root)
