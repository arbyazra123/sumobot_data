from performance_generator import generate

BASE_DIR = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
# filters = {
#     "Timer": [45.0, 60.0],
#     "ActInterval": [0.1, 0.2],
#     "Round": ["BestOf1"],
#     "SkillLeft": ["Boost"],
#     "SkillRight": ["Boost"],
# }
filters = None

matchup, bot = generate(BASE_DIR, filters)