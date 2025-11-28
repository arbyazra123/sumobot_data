"""
Configuration and constants for sumobot analyzer
"""
import numpy as np

# =====================
# Arena Configuration
# =====================
arena_center = np.array([0.24, 1.97])
arena_radius = 4.73485

# =====================
# Visualization Parameters
# =====================
tile_size = 0.7   # Larger = bigger heatmap tiles (lower resolution)

# =====================
# Bot Marker Configuration
# =====================
# Map bot names to matplotlib marker shapes for easy visual differentiation
BOT_MARKER_MAP = {
    "Bot_NN": "o",           # #1: NN - Circle
    "Bot_ML_Classification": "s",          # #2: MLP - Square
    "Bot_MCTS": "8",         # #3: MCTS - Octagon
    "Bot_FuzzyLogic": "^",        # #4: Fuzzy - Triangle up
    "Bot_Primitive": "p",    # #5: Primitive - Pentagon
    "Bot_GA": "h",           # #6: GA - Hexagon
    "Bot_SLM_ActionGPT": "*",          # #7: SLM - Star
    "Bot_PPO": "8",          # #8: PPO - Octagon
    "Bot_BT": "X",           # #9: BT - X filled
    "Bot_UtilityAI": "P",      # #10: Utility - Plus
    "Bot_LLM_ActionGPT": "D",          # #11: LLM - Diamond
    "Bot_FSM": "v",          # #12: FSM - Triangle down
    "Bot_DQN": "d",          # #13: DQN - Thin diamond
}

# Default marker if bot not in map
DEFAULT_MARKER = "o"

# =====================
# Metric Name Mapping
# =====================
# Map metric/key names to proper display names
METRIC_NAME_MAP = {
    # Time-related metrics
    "MatchDur": "Match Duration",
    "ActInterval": "Action Interval",
    "Timer": "Timer",
    "Duration": "Duration",

    # Win/Performance metrics
    "WinRate": "Win Rate",
    "WinRate_L": "Win Rate (Left)",
    "WinRate_R": "Win Rate (Right)",
    "Rank": "Rank",

    # Action metrics
    "ActionCounts": "Action Counts",
    "ActionCounts_L": "Action Counts (Left)",
    "ActionCounts_R": "Action Counts (Right)",
    "Actions": "Actions",
    "AvgActions_L": "Avg Actions (Left)",
    "AvgActions_R": "Avg Actions (Right)",

    # Collision metrics
    "Collisions": "Collisions",
    "Collisions_L": "Collisions (Left)",
    "Collisions_R": "Collisions (Right)",
    "TotalCollisions": "Total Collisions",
    "Actor_L": "Actor (Left)",
    "Actor_R": "Actor (Right)",
    "Tie": "Tie",

    # Specific action types
    "Accelerate_Act": "Accelerate",
    "Accelerate_Dur": "Accelerate",
    "Accelerate_Act_L": "Accelerate (Left)",
    "Accelerate_Act_R": "Accelerate (Right)",
    "TurnLeft_Act": "Turn Left",
    "TurnLeft_Dur": "Turn Left",
    "TurnLeft_Act_L": "Turn Left (Left)",
    "TurnLeft_Act_R": "Turn Left (Right)",
    "TurnRight_Act": "Turn Right",
    "TurnRight_Dur": "Turn Right",
    "TurnRight_Act_L": "Turn Right (Left)",
    "TurnRight_Act_R": "Turn Right (Right)",
    "Dash_Act": "Dash",
    "Dash_Dur": "Dash",

    # Skill actions
    "SkillBoost_Act": "Skill Boost",
    "SkillBoost_Dur": "Skill Boost",
    "SkillBoost_Act_L": "Skill Boost (Left)",
    "SkillBoost_Act_R": "Skill Boost (Right)",
    "SkillStone_Act": "Skill Stone",
    "SkillStone_Dur": "Skill Stone",
    "SkillStone_Act_L": "Skill Stone (Left)",
    "SkillStone_Act_R": "Skill Stone (Right)",
    "TotalSkillAct": "Total Skill Actions",

    # Round/Game metrics
    "Round": "Round",
    "RoundNumeric": "Round",
    "SkillTypeNumeric": "Skill Type",
    "Games": "Games",

    # Bot identifiers
    "Bot": "Bot",
    "Bot_L": "Bot (Left)",
    "Bot_R": "Bot (Right)",
    "Enemy": "Enemy",
    "Left_Side": "Left Side",
    "Right_Side": "Right Side",

    # Skill types
    "Skill": "Skill",
    "SkillType": "Skill Type",
    "SkillLeft": "Skill (Left)",
    "SkillRight": "Skill (Right)",
    "SkillNumeric": "Skill (Numeric)",

    # Time bins
    "TimeBin": "Time Bin",

    # Other metrics
    "AvgDuration": "Avg Duration",
    "MeanCount": "Mean Count",
    "Count": "Count",
    "Action": "Action",
    "Side": "Side",
    "BotWithRank": "Bot (with Rank)",
    "BotWithRankLeft": "Bot (Left, with Rank)",
    "BotWithRankRight": "Bot (Right, with Rank)",
}


def get_metric_name(metric_key):
    """
    Get proper display name for a metric key.

    Args:
        metric_key: Raw metric/column name

    Returns:
        Proper display name if found in map, otherwise returns the raw metric key
    """
    return METRIC_NAME_MAP.get(metric_key, metric_key)


def get_bot_marker(bot_name):
    """
    Get marker shape for a given bot name.

    Args:
        bot_name: Name of the bot

    Returns:
        Matplotlib marker string
    """
    return BOT_MARKER_MAP.get(bot_name, DEFAULT_MARKER)
