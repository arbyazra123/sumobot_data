"""
Bot marker configuration for plotting.
This module provides bot-specific marker shapes for matplotlib plots.
"""

# =====================
# Bot Marker Configuration
# =====================
# Map bot names to matplotlib marker shapes for easy visual differentiation
BOT_MARKER_MAP = {
    "Bot_BT": "o",           # Circle
    "Bot_NN": "s",           # Square
    "Bot_Primitive": "^",    # Triangle up
    "Bot_MCTS": "D",         # Diamond
    "Bot_RL": "v",           # Triangle down
    "Bot_Hybrid": "P",       # Plus (filled)
    "Bot_Random": "*",       # Star
    "Bot_Aggressive": "X",   # X
    "Bot_Defensive": "p",    # Pentagon
    "Bot_Custom": "h",       # Hexagon
}

# Default marker if bot not in map
DEFAULT_MARKER = "o"


def get_bot_marker(bot_name):
    """
    Get marker shape for a given bot name.

    Args:
        bot_name: Name of the bot

    Returns:
        Matplotlib marker string
    """
    return BOT_MARKER_MAP.get(bot_name, DEFAULT_MARKER)


def plot_with_bot_markers(ax, data, x, y, hue, hue_order=None, **kwargs):
    """
    Plot line plot with bot-specific markers.

    Args:
        ax: Matplotlib axes object
        data: pandas DataFrame with plot data
        x: Column name for x-axis
        y: Column name for y-axis
        hue: Column name for grouping (bot names or bot names with rank)
        hue_order: List specifying order of hue values (optional)
        **kwargs: Additional plot keywords (linewidth, alpha, etc.)

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_with_bot_markers(ax, data=df, x="Timer", y="WinRate",
        ...                       hue="BotWithRank", hue_order=bot_order)
    """
    # Default plot settings
    plot_kwargs = {'linewidth': 2, 'markersize': 8}
    plot_kwargs.update(kwargs)

    # Determine which bots to plot
    bots_to_plot = hue_order if hue_order else data[hue].unique()

    for bot_label in bots_to_plot:
        bot_data = data[data[hue] == bot_label]
        if bot_data.empty:
            continue

        # Extract original bot name (before " (#rank)" if present)
        bot_name = bot_label.split(" (")[0] if " (" in str(bot_label) else str(bot_label)
        marker = get_bot_marker(bot_name)

        ax.plot(bot_data[x], bot_data[y], marker=marker, label=bot_label, **plot_kwargs)


def update_bot_marker_map(new_mappings):
    """
    Update the bot marker map with new mappings.

    Args:
        new_mappings: Dictionary of {bot_name: marker_shape}

    Example:
        >>> update_bot_marker_map({"Bot_NewBot": "H"})
    """
    BOT_MARKER_MAP.update(new_mappings)
