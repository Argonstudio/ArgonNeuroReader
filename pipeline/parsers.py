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

# pipeline/parsers.py
import re
import os
import time
import hashlib
import json
from datetime import datetime
from lxml import etree
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

MAX_PARAGRAPH_LENGTH = 1500
GEMINI_CHUNK_SIZE = 20_000      # символов на один запрос
GEMINI_DELAY = 15               # секунд между запросами (увеличено для free-tier)
GEMINI_RETRY_BASE_DELAY = 5     # база для обычных ошибок
GEMINI_MAX_ATTEMPTS = 5         # максимум попыток на один фрагмент (включая 429)
MAX_CONSECUTIVE_429 = 5         # после стольких 429 подряд прекращаем обработку (дневной лимит)

os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:10808"


def _log_progress(book_dir, message):
    """Запись в лог-файл прогресса Gemini."""
    log_path = os.path.join(book_dir, 'gemini_progress.log')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except:
        pass


def clean_paragraphs(paragraphs: list[str], max_length: int = MAX_PARAGRAPH_LENGTH) -> list[str]:
    """Минимальная очистка параграфов."""
    cleaned = []
    for p in paragraphs:
        p = p.replace('\xad', '')
        p = re.sub(r'[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f-\u009f'
                   r'\u2000-\u200f\u2028-\u202f\u205f-\u206f\ufeff\ufff0-\uffff]', '', p)
        p = re.sub(r'\s+', ' ', p).strip()
        if not p:
            continue
        while len(p) > max_length:
            split_pos = p.rfind(' ', 0, max_length)
            if split_pos == -1:
                split_pos = max_length
            cleaned.append(p[:split_pos].strip())
            p = p[split_pos:].strip()
        if p:
            cleaned.append(p)
    return cleaned


# --- Gemini ---
def _get_api_keys():
    raw = os.getenv("GEMINI_API_KEY", "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _load_spaces_for_gemini(spaces_dir: str) -> str:
    """Читает файл ударений и возвращает строку для вставки в промпт."""
    if not os.path.exists(spaces_dir):
        return ""
    files = [f for f in os.listdir(spaces_dir) if os.path.isfile(os.path.join(spaces_dir, f))]
    if not files:
        return ""
    file_path = os.path.join(spaces_dir, files[0])
    encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'windows-1251', 'latin-1']
    lines = None
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if lines is None:
        return ""
    return ''.join(lines)


def _extract_retry_delay(error_message: str) -> int:
    """Извлекает рекомендованное время ожидания (сек) из ошибки 429."""
    match = re.search(r'retry in (\d+\.?\d*)s', error_message, re.IGNORECASE)
    if match:
        return int(float(match.group(1)))
    return 60


def _gemini_fix(text: str, api_key: str, tts_choice: str, spaces_content: str) -> str:
    base_prompt = ""
    if tts_choice == "edge":
        base_prompt = (
            "Ты — профессиональный редактор, специализирующийся на подготовке литературных текстов для систем озвучивания (TTS), в частности Edge-TTS.\n"
            "Твоя цель — адаптировать текст так, чтобы он звучал максимально естественно и грамотно в исполнении нейросети, сохранив при этом авторский стиль.\n\n"
            
            "ОБЩИЕ ИНСТРУКЦИИ:\n"
            "1. Анализ языка: Если исходный текст написан НЕ на русском языке, переведи его на русский литературный язык. Если текст на русском, используй его как основу для дальнейшей обработки.\n"
            "2. Очистка: Удали из текста элементы, которые не должны быть озвучены (например, служебные слова, блоки с оглавлениями, прямые ссылки на веб-сайты, если они есть в чанке).\n"
            "3. Сохранение стиля: Вноси изменения аккуратно. Не переписывай авторский стиль, не удаляй важные смысловые куски и не меняй слова местами без необходимости.\n\n"
            
            "ТЕХНИЧЕСКИЕ ПРАВИЛА ОЗВУЧКИ (EDGE-TTS): \n"
            "Для корректной работы движка соблюдай следующие ограничения:\n"
            "1. Запрещены SSML-теги (например, <speak>, <sub>, <phoneme>), специальные символы ударения (акуты) и скрытые юникод-символы.\n"
            "2. Запрещены заглавные буквы внутри слов (например, 'зАмок', 'замОк'). Используй стандартные заглавные буквы только в начале предложений и имен собственных.\n\n"
            
            "ЛИНГВИСТИЧЕСКАЯ АДАПТАЦИЯ:\n"
            "1. Работа с омографами: Слова, меняющие смысл от ударения (за́мок/замо́к, сто́ит/стои́т), должны быть адаптированы под контекст. Если движок может ошибиться, замени слово синонимом или перестрой фразу для однозначности.\n"
            "   Пример: 'я видел замок' -> 'я видел дворец' (если речь о дворце).\n"
            "2. Буква 'Ё': Всегда восстанавливай букву 'Ё', где это необходимо по смыслу и правилам русского языка, чтобы избежать неверного чтения.\n"
            "3. Числа и цифры: Заменяй арабские цифры словами в соответствующем падеже. Латиницу (названия брендов, термины) транскрибируй кириллицей строчными буквами.\n"
            "   Пример: 'в 2024 году' -> 'в две тысячи двадцать четвёртом году', 'iPhone' -> 'айфон'.\n"
            "4. Аббревиатуры и сокращения: Полностью расшифровывай сокращения (г., ул., млн) и согласовывай их падежные окончания с соседними словами.\n"
            "5. Синтаксис: Если предложение слишком длинное или сложное, разбей его на два более коротких или добавь паузы (запятые, тире), чтобы дать движку возможность сделать естественный вдох.\n"
            "6. Устаревшие слова: Заменяй редкие или архаичные слова на современные аналоги, если это не нарушает атмосферу произведения.\n\n"
            
            "ЗАЩИТА ОТ ОШИБОК И ФОРМАТ ВЫВОДА:\n"
            "Внимательно изучи входной фрагмент.\n"
            "- Если входной фрагмент пуст, состоит только из пробелов, знаков препинания или служебной информации, верни его в исходном виде.\n"
            "- В остальных случаях верни обработанный текст единым блоком, сохраняя исходное разбиение на абзацы.\n"
            "- **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:** Использовать markdown-блоки (знаки ```), добавлять любые комментарии, пояснения от редактора или заголовки к ответу. Выводи только текст.\n\n"
            
            "Фрагмент для обработки:\n{text}"
        )
    else:  # silero
        base_prompt = (
            "Ты — профессиональный корректор русской литературы. "
            "Очисти текст от дефектов PDF-конвертации. "
            "Найди в тексте слова-омографы, значение которых зависит от ударения "
            "(например: замок, стоит, белки, пробы), а также сложные научные/биологические термины, "
            "в которых может ошибиться Silero TTS. "
            "Поставь знак плюс «+» строго ПЕРЕД ударной гласной только в этих словах. "
            "Примеры: «зам+ок», «з+амок», «сто+ит», «ст+оит», «б+елки». "
            "В простых и однозначных словах знаки плюса НЕ СТАВЬ, чтобы не перегружать движок. К примеру 'м+олодость' или 'челов+ек'. "
            "Фамилии и имена не трогай"
            "Замени римские цифры и обычные цифры на слова. К примеру 'XIX век' на 'Девятнадцатый век'. Еще один пример 'Глава 1' на 'Глава первая'. Это очень важно."
        )
    
    if spaces_content.strip():
        spaces_instruction = (
            "\n\nКроме того, у меня есть файл со списком слов, в которых обязательно нужно расставить ударения. "
            "Примени эти ударения в итоговом тексте в первую очередь, даже если они противоречат твоему анализу. "
            "Важное правило форматирования: в исходном списке ударная гласная отмечена знаком ( ́ ). "
            "Тебе нужно полностью удалить этот знак и изменить формат под целевой движок озвучки: "
            "1) Если делаешь версию для Edge — Вставляй специальный символ комбинируемого акута ́ (U+0301) строго после ударной гласной. "
            "2) Если делаешь версию для Silero — поставь знак плюс ПЕРЕД ударной гласной (например: челов+ек, гиппок+амп). "
            "Учитывай падежи и числа слов в тексте, перенося выбранный формат на ту же гласную корня или основы. "
            "Вот список слов/фраз с ударениями:\n"
            f"{spaces_content.strip()}"
        )
        prompt = base_prompt + spaces_instruction + "\n\nВерни только готовый очищенный текст без твоих комментариев.\n\n" + f"Текст для исправления:\n{text}"
    else:
        prompt = base_prompt + "\n\nВерни только готовый очищенный текст без твоих комментариев.\n\n" + f"Текст для исправления:\n{text}"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _get_cache_path(book_path: str, tts_choice: str) -> str:
    """Путь к ИТОГОВОМУ кэш-файлу (для обратной совместимости)."""
    cache_dir = os.path.join(os.path.dirname(book_path), "gemini_cache")
    os.makedirs(cache_dir, exist_ok=True)
    hash_input = book_path + tts_choice
    hash_hex = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hash_hex}.txt")


def _get_chunks_dir(book_path: str, tts_choice: str) -> str:
    """Папка для инкрементальных чанков."""
    base = os.path.dirname(book_path)
    book_name = os.path.splitext(os.path.basename(book_path))[0]
    chunks_dir = os.path.join(base, "gemini_chunks", f"{book_name}_{tts_choice}")
    os.makedirs(chunks_dir, exist_ok=True)
    return chunks_dir


def _fix_with_gemini(paragraphs: list[str], tts_choice: str, book_path: str) -> list[str]:
    api_keys = _get_api_keys()
    if not api_keys:
        print("Не задан GEMINI_API_KEY. Gemini-исправление пропущено.")
        return paragraphs

    book_dir = os.path.dirname(book_path)
    final_cache_path = _get_cache_path(book_path, tts_choice)
    chunks_dir = _get_chunks_dir(book_path, tts_choice)

    # Если итоговый кэш уже есть – используем его
    if os.path.exists(final_cache_path):
        _log_progress(book_dir, f"Найден итоговый кэш Gemini: {final_cache_path}")
        with open(final_cache_path, 'r', encoding='utf-8') as f:
            cached_text = f.read()
        marker = "\n<<<PARA>>>\n"
        return [p.strip() for p in cached_text.split(marker) if p.strip()]

    # Загружаем spaces
    spaces_dir = os.path.join(book_dir, '..', 'spaces')
    spaces_content = _load_spaces_for_gemini(spaces_dir)

    marker = "\n<<<PARA>>>\n"
    full_text = marker.join(paragraphs)

    # Разбиваем на чанки
    chunks = []
    remaining = full_text
    while len(remaining) > GEMINI_CHUNK_SIZE:
        split_pos = remaining.rfind(' ', 0, GEMINI_CHUNK_SIZE)
        if split_pos == -1:
            split_pos = GEMINI_CHUNK_SIZE
        chunks.append(remaining[:split_pos].strip())
        remaining = remaining[split_pos:].strip()
    if remaining:
        chunks.append(remaining)

    total_chunks = len(chunks)
    _log_progress(book_dir, f"Начало обработки: всего {total_chunks} чанков для книги {os.path.basename(book_path)}")

    # Проверяем, какие чанки уже сохранены
    processed_indices = set()
    for fname in os.listdir(chunks_dir):
        if fname.startswith("chunk_") and fname.endswith(".txt"):
            try:
                idx = int(fname[6:10])  # chunk_XXXX.txt
                processed_indices.add(idx)
            except:
                pass
    if processed_indices:
        _log_progress(book_dir, f"Найдены готовые чанки: {sorted(processed_indices)}")

    fixed_chunks = []
    for i, chunk in enumerate(chunks, 1):
        chunk_file = os.path.join(chunks_dir, f"chunk_{i:04d}.txt")
        if i in processed_indices:
            # Читаем готовый чанк
            with open(chunk_file, 'r', encoding='utf-8') as f:
                fixed_text = f.read().strip()
            fixed_chunks.append(fixed_text)
            _log_progress(book_dir, f"Чанк {i}/{total_chunks} загружен из кэша")
            continue

        # Иначе обрабатываем через Gemini
        key = api_keys[(i - 1) % len(api_keys)]
        _log_progress(book_dir, f"Отправка чанка {i}/{total_chunks} ({len(chunk)} символов) с ключом ...{key[-8:]}")

        fixed = None
        attempt = 0
        consecutive_429 = 0
        while attempt < GEMINI_MAX_ATTEMPTS:
            attempt += 1
            try:
                fixed = _gemini_fix(chunk, api_key=key, tts_choice=tts_choice, spaces_content=spaces_content)
                # Успех – сбрасываем счётчик 429
                consecutive_429 = 0
                break
            except Exception as e:
                err_msg = str(e)
                if '429' in err_msg or 'ResourceExhausted' in err_msg:
                    consecutive_429 += 1
                    retry_delay = _extract_retry_delay(err_msg)
                    _log_progress(book_dir, f"  !!! 429 Rate Limit (попытка {consecutive_429} подряд). Ожидание {retry_delay} сек...")
                    if consecutive_429 >= MAX_CONSECUTIVE_429:
                        _log_progress(book_dir, "Дневной лимит исчерпан. Сохраняем прогресс и останавливаемся.")
                        # Сохраняем то, что уже обработано, в итоговый кэш (частичный)
                        partial_fixed = marker.join(fixed_chunks + [chunk])  # последний чанк остаётся оригиналом
                        with open(final_cache_path, 'w', encoding='utf-8') as f:
                            f.write(partial_fixed)
                        # Удаляем итоговый кэш, чтобы при следующем запуске не подумали, что всё готово
                        os.remove(final_cache_path)
                        # Возвращаем то, что есть (с оригинальным текстом для необработанных чанков)
                        result = [p.strip() for p in partial_fixed.split(marker) if p.strip()]
                        _log_progress(book_dir, f"Обработка прервана. Обработано {len(fixed_chunks)} из {total_chunks} чанков.")
                        return result
                    time.sleep(retry_delay)
                    attempt -= 1  # не тратим попытку
                else:
                    consecutive_429 = 0
                    _log_progress(book_dir, f"  [Попытка {attempt}/{GEMINI_MAX_ATTEMPTS}] Ошибка: {e}")
                    if attempt < GEMINI_MAX_ATTEMPTS:
                        sleep_time = GEMINI_RETRY_BASE_DELAY * attempt
                        time.sleep(sleep_time)
                    else:
                        _log_progress(book_dir, f"  !!! Все попытки исчерпаны для чанка {i}. Используется оригинальный текст.")
                        fixed = chunk

        if fixed is None:
            fixed = chunk   # подстраховка

        # Сохраняем обработанный чанк немедленно
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(fixed)
        _log_progress(book_dir, f"Чанк {i}/{total_chunks} сохранён в {chunk_file}")
        fixed_chunks.append(fixed)

        # Задержка между чанками
        if i < total_chunks:
            time.sleep(GEMINI_DELAY)

    # Все чанки обработаны – собираем итоговый кэш
    full_fixed = marker.join(fixed_chunks)
    with open(final_cache_path, 'w', encoding='utf-8') as f:
        f.write(full_fixed)
    _log_progress(book_dir, f"Итоговый кэш сохранён: {final_cache_path}")
    return [p.strip() for p in full_fixed.split(marker) if p.strip()]


# --- Парсеры ---
def parse_fb2(file_path: str, tts_choice: str = None) -> list[str]:
    with open(file_path, 'rb') as f:
        tree = etree.parse(f)
    paragraphs = []
    for p in tree.xpath('//*[local-name()="p"]'):
        text = ''.join(p.itertext()).strip()
        if text:
            clean = re.sub(r'\s+', ' ', text)
            paragraphs.append(clean)
    paragraphs = clean_paragraphs(paragraphs)
    if tts_choice:
        print(f"Исправление текста FB2 с помощью Gemini ({tts_choice})...")
        paragraphs = _fix_with_gemini(paragraphs, tts_choice, file_path)
        paragraphs = clean_paragraphs(paragraphs)
    return paragraphs


def parse_txt(file_path: str, tts_choice: str = None) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = re.split(r'\n\s*\n', content)
    paragraphs = []
    for part in parts:
        clean = re.sub(r'\s+', ' ', part).strip()
        if clean:
            paragraphs.append(clean)
    paragraphs = clean_paragraphs(paragraphs)
    if tts_choice:
        print(f"Исправление текста TXT с помощью Gemini ({tts_choice})...")
        paragraphs = _fix_with_gemini(paragraphs, tts_choice, file_path)
        paragraphs = clean_paragraphs(paragraphs)
    return paragraphs


def parse_pdf(file_path: str, page_mode: str = 'all', pages=None, tts_choice: str = None) -> list[str]:
    try:
        import pdfplumber
    except ImportError:
        print("Ошибка: pdfplumber не установлен.")
        return []

    paragraphs = []
    HEADER_MARGIN = 100
    FOOTER_MARGIN = 100

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        if page_mode == 'all':
            page_nums = list(range(total_pages))
        elif page_mode == 'list':
            if pages is None:
                page_nums = list(range(total_pages))
            else:
                excluded = set(p - 1 for p in pages if 1 <= p <= total_pages)
                page_nums = [i for i in range(total_pages) if i not in excluded]
        elif page_mode == 'range':
            if pages is None:
                page_nums = list(range(total_pages))
            else:
                start, end = pages
                page_nums = [i - 1 for i in range(max(1, start), min(end, total_pages) + 1)]
        else:
            return []

        for num in page_nums:
            page = pdf.pages[num]
            cropped = page.within_bbox((0, HEADER_MARGIN, page.width, page.height - FOOTER_MARGIN))
            text = cropped.extract_text()
            if not text:
                continue
            for part in re.split(r'\n\s*\n', text):
                clean = re.sub(r'\s+', ' ', part).strip()
                if clean:
                    paragraphs.append(clean)

    paragraphs = clean_paragraphs(paragraphs)
    if tts_choice:
        print(f"Исправление текста PDF с помощью Gemini ({tts_choice})...")
        paragraphs = _fix_with_gemini(paragraphs, tts_choice, file_path)
        paragraphs = clean_paragraphs(paragraphs)
    return paragraphs