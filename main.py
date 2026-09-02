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

import os
import asyncio
import glob
from datetime import datetime

# Humanize-функции, не требующие CUDA
from humanize_final import humanize_audio as humanize_audio_std
from humanize_final_aidar import humanize_audio as humanize_audio_aidar
# humanize_silero импортируется только при выборе Silero

from pipeline.config import *
from pipeline.common import (
    log_message, load_state, save_state,
    merge_wav_files, merge_to_mp3,
    wait_for_free_vram, process_rvc_block
)
from pipeline.parsers import parse_fb2, parse_txt, parse_pdf


def choose_tts():
    print("\n=== ВЫБОР TTS ===")
    print("1 - Edge TTS (интернет, голос Дмитрий)")
    print("2 - Silero TTS (локально, версия v5)")
    choice = input("Ваш выбор (1/2): ").strip()
    if choice == "1":
        return ("edge", None, 1.0)
    elif choice == "2":
        print("\nВыберите голос Silero v5:")
        print("1 - Eugene (мужской, речь замедляется на 10%)")
        print("2 - Aidar (мужской, речь без изменения скорости)")
        voice_choice = input("Ваш выбор (1/2): ").strip()
        if voice_choice == "1":
            return ("silero", "eugene", 0.9)
        elif voice_choice == "2":
            return ("silero", "aidar", 1.0)
        else:
            print("Некорректный ввод, выбран Eugene по умолчанию")
            return ("silero", "eugene", 0.9)
    else:
        print("Некорректный ввод, выбран Edge по умолчанию")
        return ("edge", None, 1.0)


def ask_humanize():
    print("\n=== ПРИМЕНЯТЬ HUMANIZE (финальное очеловечивание) ===")
    print("1 - Да, применять мягкую обработку голоса (рекомендуется)")
    print("2 - Нет, пропустить (только RVC → сразу склейка)")
    while True:
        choice = input("Ваш выбор (1/2): ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Некорректный ввод, введите 1 или 2.")


def ask_gemini():
    print("\n=== КОРРЕКЦИЯ ТЕКСТА ЧЕРЕЗ GEMINI ===")
    print("1 - Да, исправить опечатки, омографы и раскрыть числа (требуется API-ключ)")
    print("2 - Нет, использовать оригинальный текст книги без изменений")
    while True:
        choice = input("Ваш выбор (1/2): ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Некорректный ввод, введите 1 или 2.")


async def main():
    # Динамические импорты pipeline
    from pipeline.edge_tts import pipeline_edge, get_or_create_fragments
    from pipeline.silero_tts import process_silero, prepare_groups_silero

    print("\n=== ПОЛНАЯ ОБРАБОТКА КНИГИ ===")
    tts_choice, silero_speaker, stretch_rate = choose_tts()
    do_humanize = ask_humanize()
    do_gemini = ask_gemini()  # Запрашиваем выбор у пользователя

    # Если пользователь отказался от Gemini, передаем None в парсеры
    parser_tts = tts_choice if do_gemini else None

    # Поиск книги (FB2, TXT или PDF)
    fb2_files = glob.glob(os.path.join(BOOK_DIR, "*.fb2"))
    txt_files = glob.glob(os.path.join(BOOK_DIR, "*.txt"))
    pdf_files = glob.glob(os.path.join(BOOK_DIR, "*.pdf"))

    if fb2_files:
        book_format = 'fb2'
        book_path = fb2_files[0]
    elif txt_files:
        book_format = 'txt'
        book_path = txt_files[0]
    elif pdf_files:
        book_format = 'pdf'
        book_path = pdf_files[0]
    else:
        print("!!! ОШИБКА: Положите .fb2, .txt или .pdf файл в папку book")
        return

    book_name = os.path.splitext(os.path.basename(book_path))[0]
    book_dir = os.path.join(BASE_DIR, 'output_audio', book_name)

    tts_out = os.path.join(book_dir, 'parts')
    rvc_out = os.path.join(book_dir, 'rvc_ready')
    humanized_out = os.path.join(book_dir, 'humanized_parts')
    final_out = os.path.join(book_dir, 'final_hours')

    for d in [tts_out, rvc_out, final_out]:
        os.makedirs(d, exist_ok=True)
    if do_humanize:
        os.makedirs(humanized_out, exist_ok=True)

    with open(os.path.join(book_dir, 'process.log'), 'a', encoding='utf-8') as lf:
        lf.write(f"\n--- Запуск {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    # Парсинг книги с Gemini-коррекцией (или без неё)
    if book_format == 'fb2':
        paragraphs = parse_fb2(book_path, tts_choice=parser_tts)
    elif book_format == 'txt':
        paragraphs = parse_txt(book_path, tts_choice=parser_tts)
    else:  # PDF
        print("\nНастройка извлечения страниц PDF:")
        print("1 - Все страницы")
        print("2 - Исключить определённые страницы (введите номера через пробел)")
        print("3 - Только диапазон страниц (начало и конец через пробел)")
        page_choice = input("Ваш выбор (1/2/3): ").strip()
        if page_choice == '1':
            paragraphs = parse_pdf(book_path, 'all', tts_choice=parser_tts)
        elif page_choice == '2':
            pages_str = input("Введите номера исключаемых страниц через пробел: ").strip()
            pages = [int(x) for x in pages_str.split() if x.isdigit()]
            paragraphs = parse_pdf(book_path, 'list', pages, tts_choice=parser_tts)
        elif page_choice == '3':
            range_str = input("Введите начало и конец диапазона через пробел: ").strip()
            parts = range_str.split()
            if len(parts) >= 2:
                start, end = int(parts[0]), int(parts[1])
                paragraphs = parse_pdf(book_path, 'range', (start, end), tts_choice=parser_tts)
            else:
                print("Некорректный ввод, используются все страницы.")
                paragraphs = parse_pdf(book_path, 'all', tts_choice=parser_tts)
        else:
            print("Некорректный ввод, используются все страницы.")
            paragraphs = parse_pdf(book_path, 'all', tts_choice=parser_tts)

        if not paragraphs:
            print("Не удалось извлечь текст из PDF.")
            return

    if not paragraphs:
        print("Не удалось извлечь текст из файла.")
        return

    if tts_choice == "edge":
        fragments = get_or_create_fragments(
            paragraphs, book_dir,
            chunk_min=EDGE_CHUNK_SYMBOLS_MIN,
            chunk_max=EDGE_CHUNK_SYMBOLS_MAX,
            log_message=log_message
        )
        await pipeline_edge(
            book_dir=book_dir,
            groups=fragments,
            tts_out=tts_out,
            rvc_out=rvc_out,
            humanized_out=humanized_out,
            final_out=final_out,
            do_humanize=do_humanize,
            edge_speaker=EDGE_SPEAKER,
            sample_rate=SAMPLE_RATE,
            edge_delay_min=EDGE_DELAY_MIN,
            edge_delay_max=EDGE_DELAY_MAX,
            edge_parallel=EDGE_PARALLEL,
            target_chunk_sec=TARGET_CHUNK_SEC,
            vram_free_ratio=VRAM_FREE_RATIO,
            vram_check_interval=VRAM_CHECK_INTERVAL,
            humanize_func=humanize_audio_std,
            merge_wav_files=merge_wav_files,
            process_rvc_block=process_rvc_block,
            wait_for_free_vram=wait_for_free_vram,
            merge_to_mp3=merge_to_mp3,
            load_state=load_state,
            save_state=save_state,
            log_message=log_message
        )
    else:  # silero
        try:
            from humanize_silero import humanize_silero
        except ImportError as e:
            print(f"Ошибка импорта humanize_silero: {e}")
            print("Убедитесь, что установлен CUDA Toolkit и CuPy. Невозможно продолжить с Silero.")
            return

        groups = prepare_groups_silero(paragraphs, SILERO_CHUNK_SYMBOLS)
        process_silero(
            book_dir=book_dir,
            groups=groups,
            tts_out=tts_out,
            rvc_out=rvc_out,
            humanized_out=humanized_out,
            final_out=final_out,
            do_humanize=do_humanize,
            silero_speaker=silero_speaker,
            stretch_rate=stretch_rate,
            rvc_python=RVC_PYTHON,
            rvc_cli=RVC_CLI,
            model_name=MODEL_NAME,
            index_path=INDEX_PATH,
            default_index_rate=DEFAULT_INDEX_RATE,
            is_half=IS_HALF,
            humanize_func=humanize_audio_aidar,
            humanize_silero_func=humanize_silero,
            merge_to_mp3=merge_to_mp3,
            log_message=log_message,
            sample_rate=SAMPLE_RATE
        )

    log_message(f"Генерация книги завершена. Результаты в {final_out}", book_dir)
    print(f"\nРезультаты в папке: {final_out}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
    
