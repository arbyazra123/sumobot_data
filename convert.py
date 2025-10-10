import time
from performance_generator import batch, generate

BASE_DIR = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
# filters = {
#     "Timer": [45.0],
#     "ActInterval": [0.1],
#     "Round": ["BestOf1"],
#     "SkillLeft": ["Boost"],
#     "SkillRight": ["Boost"],
# }
filters = None

batch(BASE_DIR, filters)
matchup, bot = generate()