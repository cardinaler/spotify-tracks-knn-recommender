import os
import shutil
import kagglehub

# Загрузка датасета в стандартный кэш
cache_path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

# Желаемый путь
target_path = "./data/spotify_data"

# Копирование файлов в желаемый путь
shutil.copytree(cache_path, target_path, dirs_exist_ok=True)

print("Файлы датасета скопированы в:", os.path.abspath(target_path))