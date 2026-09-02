"""
ArgonNeuroReader
Copyright (c) 2026 Ivan Voitkov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Ivan Voitkov
Website: https://argon-studio.ru/
GitHub: https://github.com/Argonstudio/ArgonNeuroReader
"""

# pipeline/config.py
import os

# Базовые пути
BASE_DIR = r'C:\newApp\books\audio_project' #поменять на путь к папке проекта
BOOK_DIR = os.path.join(BASE_DIR, 'book')
RVC_DIR = os.path.join(BASE_DIR, 'rvc_engine')
RVC_CLI = os.path.join(RVC_DIR, 'tools', 'infer_cli.py')
RVC_PYTHON = os.path.join(RVC_DIR, 'venv310', 'Scripts', 'python.exe')

# Общие настройки аудио
SAMPLE_RATE = 48000

# Edge-TTS
EDGE_SPEAKER = "ru-RU-DmitryNeural"
EDGE_CHUNK_SYMBOLS_MIN = 900
EDGE_CHUNK_SYMBOLS_MAX = 3000
EDGE_DELAY_MIN = 0.5
EDGE_DELAY_MAX = 3.0
EDGE_PARALLEL = 5

# Silero
SILERO_CHUNK_SYMBOLS = 12000

# RVC
DEFAULT_INDEX_RATE = 0.80
MODEL_NAME = "voiceNew2_e290_s28130.pth" #поменять имя модели
INDEX_PATH = os.path.join(RVC_DIR, "logs", "voiceNew2",
                          "added_IVF1432_Flat_nprobe_1_voiceNew2_v2.index") #поменять index файл модели
IS_HALF = "True"

# Склейка и VRAM
TARGET_CHUNK_SEC = 10 * 60          # 10 минут
VRAM_FREE_RATIO = 0.7
VRAM_CHECK_INTERVAL = 600

# Целевая длительность одного итогового MP3 (сек)
FINAL_MP3_DURATION_SEC = 60 * 60   # 1 час
