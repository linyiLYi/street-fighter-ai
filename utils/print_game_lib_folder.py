import os

import retro

retro_directory = os.path.dirname(retro.__file__)
game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis"
print(os.path.join(retro_directory, game_dir))
