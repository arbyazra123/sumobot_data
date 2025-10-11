import time
from generator import batch, generate
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
chunksize = 10_000

batch(BASE_DIR, filters, batch_size, chunksize)
# batch_parquet(BASE_DIR, filters, batch_size, chunksize)
matchup, bot = generate(is_parquet=False)