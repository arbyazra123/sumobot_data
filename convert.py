import time
from performance_generator import batch, generate
from parquet_generator import batch_parquet

BASE_DIR = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
# filters = {
#     "Timer": [45.0],
#     "ActInterval": [0.1],
#     "Round": ["BestOf1"],
#     "SkillLeft": ["Boost"],
#     "SkillRight": ["Boost"],
# }
filters = None
batch_size = 5

# batch(BASE_DIR, filters, batch_size)
batch_parquet(BASE_DIR, filters, batch_size)
matchup, bot = generate(is_parquet=True)