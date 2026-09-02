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
BASE_DIR = r'C:\newApp\books\audio_project'                                 # поменять на путь к папке проекта
BOOK_DIR = os.path.join(BASE_DIR, 'book')                                   # откуда брать книгу для озвучки, название папки
RVC_DIR = os.path.join(BASE_DIR, 'rvc_engine')                              # папка где RVC
RVC_CLI = os.path.join(RVC_DIR, 'tools', 'infer_cli.py')
RVC_PYTHON = os.path.join(RVC_DIR, 'venv310', 'Scripts', 'python.exe')

# Общие настройки аудио

# Частота дискретизации (Гц) для всех этапов обработки.
# 48000 Гц — стандарт для TTS и RVC, обеспечивает высокое качество.
# Не рекомендуется менять без необходимости.
SAMPLE_RATE = 48000

# Edge-TTS
EDGE_SPEAKER = "ru-RU-DmitryNeural"

# Минимальный размер текстового фрагмента (в символах) для одного запроса к Edge TTS.
# Меньшие значения — больше запросов, но стабильнее при сбоях.
# 900 символов ≈ 1.5–2 минуты речи.
EDGE_CHUNK_SYMBOLS_MIN = 900

# Максимальный размер текстового фрагмента (в символах) для одного запроса.
# Большие фрагменты экономят запросы, но повышают риск ошибки NoAudioReceived.
# 3000 символов ≈ 4–6 минут речи.
EDGE_CHUNK_SYMBOLS_MAX = 3000

# Минимальная пауза между успешными запросами к Edge TTS (в секундах).
# Слишком маленькая пауза может привести к блокировке IP.
EDGE_DELAY_MIN = 0.5

# Максимальная пауза между успешными запросами (в секундах).
# Случайная задержка в диапазоне [MIN, MAX] имитирует поведение человека.
EDGE_DELAY_MAX = 3.0

# Количество параллельных запросов к Edge TTS.
# Больше — быстрее, но выше риск блокировки и перегрузки сети.
# Рекомендуется 3–5 для стабильной работы.
EDGE_PARALLEL = 5

# ============================================================
# НАСТРОЙКИ SILERO TTS (локальный синтез речи)
# ============================================================

# Размер текстового блока (в символах) для одного вызова Silero.
# Silero обрабатывает длинные тексты лучше, чем Edge, поэтому лимит больше.
# 12000 символов ≈ 15–20 минут речи.
# Уменьшите, если возникают ошибки памяти или качества.
SILERO_CHUNK_SYMBOLS = 12000

# ============================================================
# НАСТРОЙКИ RVC (Retrieval-based Voice Conversion)
# ============================================================

# Сила влияния индексного файла при преобразовании голоса.
# Диапазон: 0.0 – 1.0
#   0.0 — минимальное влияние (голос ближе к исходному TTS)
#   1.0 — максимальное влияние (голос ближе к целевой модели)
# 0.75–0.85 обычно даёт наиболее естественный результат.
DEFAULT_INDEX_RATE = 0.80

# Имя файла обученной RVC-модели.
# Файл должен находиться в папке: RVC_DIR/logs/<имя_модели>/
# ЗАМЕНИТЕ на имя вашей модели после обучения.
# Пример: "my_voice_e300_s2500.pth"
MODEL_NAME = "voiceNew2_e290_s28130.pth" 

# Путь к индексному файлу модели RVC.
# Индекс ускоряет поиск ближайших характеристик голоса.
# Создаётся автоматически при обучении (файл .index).
# ЗАМЕНИТЕ на путь к вашему индексному файлу.
INDEX_PATH = os.path.join(RVC_DIR, "logs", "voiceNew2",
                          "added_IVF1432_Flat_nprobe_1_voiceNew2_v2.index") 

# Использовать half-precision (float16) для вычислений.
# "True"  — экономит VRAM (~40%), может незначительно снизить качество.
# "False" — точнее, но требует больше видеопамяти.
# Для GPU с 6 ГБ VRAM рекомендуется "True".
IS_HALF = "True"

# ============================================================
# НАСТРОЙКИ СКЛЕЙКИ И УПРАВЛЕНИЯ VRAM
# ============================================================

# Целевая длительность аудиоблока (в секундах) перед обработкой RVC.
# 600 секунд = 10 минут.
# Блоки по 10 минут удобны для контроля прогресса и восстановления после сбоев.
# При уменьшении — больше файлов, но меньше риск переполнения VRAM.
TARGET_CHUNK_SEC = 10 * 60          # 10 минут

# Порог свободной видеопамяти (доля от общего объёма) для запуска RVC.
# 0.7 = 70% VRAM должно быть свободно.
# Если свободной памяти меньше, программа будет ждать (см. VRAM_CHECK_INTERVAL).
# Увеличьте до 0.8–0.9, если RVC падает из-за нехватки памяти.
VRAM_FREE_RATIO = 0.7

# Интервал проверки свободной VRAM (в секундах).
# 600 секунд = 10 минут.
# Программа будет проверять доступность памяти каждые 10 минут,
# если условие VRAM_FREE_RATIO не выполнено.
VRAM_CHECK_INTERVAL = 600

