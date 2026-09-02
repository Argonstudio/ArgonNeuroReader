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


# pipeline/edge_tts.py
import os
import sys

# Защита от циклического импорта/затенения библиотеки edge-tts, 
# если директория pipeline случайно оказалась в sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p and os.path.abspath(p) != current_dir]

import asyncio
import time
import random
import json
import subprocess
import re
import threading
import glob
import soundfile as sf
import librosa
import edge_tts  # Теперь гарантированно импортирует оригинальный пакет pip edge-tts

from pipeline.common import clean_edge_audio

# ------------------------------------------------------------
# Глобальный флаг для последовательного RVC (потокобезопасный)
# ------------------------------------------------------------
rvc_busy = threading.Event()
rvc_busy.clear()   # изначально свободен

# ------------------------------------------------------------
# Генерация одного фрагмента Edge TTS (без SSML)
# ------------------------------------------------------------
async def generate_one_fragment(text, idx, tts_out, semaphore, book_dir,
                                speaker, sample_rate,
                                delay_min, delay_max, log_message):
    async with semaphore:
        f_p = os.path.join(tts_out, f"part_{idx:04d}.wav")
        if os.path.exists(f_p) and os.path.getsize(f_p) > 1000:
            try:
                dur = librosa.get_duration(path=f_p)
                log_message(f"Фрагмент {idx} уже существует, дл. {dur:.1f} сек", book_dir)
                return {'index': idx, 'path': f_p, 'duration': dur, 'success': True}
            except:
                log_message(f"Фрагмент {idx} повреждён, будет перегенерирован", book_dir)

        if not text or not text.strip():
            log_message(f"Фрагмент {idx}: пустой текст, синтез невозможен", book_dir)
            sys.exit(1)

        attempt = 0
        delay = 2.0
        max_delay = 60.0
        start_time = time.perf_counter()
        connection_errors = 0
        no_audio_errors = 0

        while True:
            attempt += 1
            try:
                comm = edge_tts.Communicate(text, speaker, proxy=None)
                await comm.save(f_p)

                if os.path.exists(f_p) and os.path.getsize(f_p) > 1000:
                    y, sr = librosa.load(f_p, sr=None)
                    y_48k = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                    sf.write(f_p, y_48k, sample_rate, subtype='PCM_16')
                    dur = librosa.get_duration(path=f_p)
                    elapsed = time.perf_counter() - start_time
                    log_message(f"Фрагмент {idx} за {elapsed:.1f} сек, дл. {dur:.1f} сек", book_dir)
                    sleep_sec = random.uniform(delay_min, delay_max)
                    log_message(f"Пауза {sleep_sec:.1f} сек", book_dir)
                    await asyncio.sleep(sleep_sec)
                    return {'index': idx, 'path': f_p, 'duration': dur, 'success': True}
                else:
                    raise Exception("Файл слишком маленький")
            except Exception as e:
                err_type = type(e).__name__
                log_message(
                    f"Фрагмент {idx} ошибка (попытка {attempt}): {err_type}: {e}. "
                    f"Текст (первые 200): {text[:200]}", book_dir)

                if 'NoAudioReceived' in err_type:
                    no_audio_errors += 1
                    if no_audio_errors >= 3:
                        log_message(
                            f"ФАТАЛЬНО: Фрагмент {idx} — превышено количество ошибок 'NoAudioReceived' (3). Завершение.",
                            book_dir)
                        sys.exit(1)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, max_delay)
                    connection_errors = 0
                elif 'Timeout' in err_type or 'Connection' in err_type:
                    connection_errors += 1
                    if connection_errors >= 3:
                        log_message(
                            f"ФАТАЛЬНО: Фрагмент {idx} — превышено количество сетевых ошибок (3). Завершение.",
                            book_dir)
                        sys.exit(1)
                    log_message(f"Сетевая ошибка. Ожидание 5 сек перед пересозданием подключения...", book_dir)
                    await asyncio.sleep(5)
                    delay = 2.0
                else:
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, max_delay)
                    connection_errors = 0
                    no_audio_errors = 0


# ------------------------------------------------------------
# Разбивка текста на фрагменты
# ------------------------------------------------------------
def get_or_create_fragments(paragraphs, book_dir, chunk_min, chunk_max, log_message):
    fragments_path = os.path.join(book_dir, 'edge_fragments.json')
    if os.path.exists(fragments_path):
        with open(fragments_path, 'r', encoding='utf-8') as f:
            fragments = json.load(f)
        log_message(f"Загружена разбивка из {fragments_path} ({len(fragments)} фрагментов)", book_dir)
        return fragments

    log_message(f"Создание новой разбивки (символов от {chunk_min} до {chunk_max})", book_dir)
    fragments = []
    curr, c_len = [], 0
    current_limit = random.randint(chunk_min, chunk_max)
    for p in paragraphs:
        if curr and c_len + len(p) > current_limit:
            fragments.append(" ".join(curr))
            curr = [p]
            c_len = len(p)
            current_limit = random.randint(chunk_min, chunk_max)
        else:
            curr.append(p)
            c_len += len(p)
    if curr:
        fragments.append(" ".join(curr))

    with open(fragments_path, 'w', encoding='utf-8') as f:
        json.dump(fragments, f, ensure_ascii=False, indent=2)
    log_message(f"Создано {len(fragments)} фрагментов, сохранено в {fragments_path}", book_dir)
    return fragments


# ------------------------------------------------------------
# Главный конвейер Edge TTS
# ------------------------------------------------------------
async def pipeline_edge(book_dir, groups, tts_out, rvc_out, humanized_out, final_out,
                        do_humanize,
                        edge_speaker, sample_rate,
                        edge_delay_min, edge_delay_max,
                        edge_parallel,
                        target_chunk_sec,
                        vram_free_ratio, vram_check_interval,
                        humanize_func,
                        merge_wav_files, process_rvc_block,
                        wait_for_free_vram, merge_to_mp3,
                        load_state, save_state, log_message):
    total_fragments = len(groups)
    log_message(
        f"Всего фрагментов: {total_fragments}, параллелизм {edge_parallel}, пауза {edge_delay_min}-{edge_delay_max} сек",
        book_dir)

    state = load_state(book_dir)
    processed_fragments = set(state.get('processed_fragments', []))
    processed_chunks = set(state.get('processed_chunks', []))

    current_batch_indices = state.get('partial_batch_indices', [])
    current_batch_dur = state.get('partial_batch_dur', 0.0)
    if not current_batch_indices and state.get('partial_batch_files'):
        for f in state['partial_batch_files']:
            try:
                idx = int(os.path.basename(f).split('_')[1].split('.')[0])
                current_batch_indices.append(idx)
            except:
                pass

    all_processed = set(processed_fragments)
    for idx in current_batch_indices:
        all_processed.add(str(idx))
    next_frag = 1
    while next_frag <= total_fragments and str(next_frag) in all_processed:
        next_frag += 1

    edge_chunks_dir = os.path.join(book_dir, 'edge_chunks')
    os.makedirs(edge_chunks_dir, exist_ok=True)

    chunk_counter = len(processed_chunks) + 1
    rvc_queue = asyncio.Queue()
    stop_worker = False

    # Воркер последовательной обработки RVC
    async def rvc_worker():
        while not stop_worker or not rvc_queue.empty():
            try:
                task = await asyncio.wait_for(rvc_queue.get(), timeout=1.0)
                chunk_name, chunk_path, file_list, expected_dur, indices = task

                if chunk_name in processed_chunks:
                    log_message(f"Воркер: блок {chunk_name} уже обработан, пропускаю", book_dir)
                    rvc_queue.task_done()
                    continue

                log_message(f"Воркер: начинаю обработку блока {chunk_name}", book_dir)

                if not merge_wav_files(file_list, chunk_path):
                    log_message(f"ФАТАЛЬНО: ошибка склейки блока {chunk_name}", book_dir)
                    sys.exit(1)

                try:
                    cleaned_path = clean_edge_audio(chunk_path)
                    chunk_path = cleaned_path
                    log_message(f"Артефакты Edge-TTS удалены для {chunk_name}", book_dir)
                except Exception as e:
                    log_message(f"Не удалось очистить аудио для {chunk_name}: {e}", book_dir)

                while rvc_busy.is_set():
                    log_message(f"Воркер: RVC занят, ожидаю до 3 минут...", book_dir)
                    rvc_busy.wait(timeout=180)
                rvc_busy.set()
                log_message(f"Воркер: RVC захвачен, запускаю обработку блока {chunk_name}", book_dir)

                try:
                    await wait_for_free_vram(book_dir, vram_free_ratio, vram_check_interval)
                    loop = asyncio.get_running_loop()
                    result_path = await loop.run_in_executor(None, process_rvc_block,
                                                             chunk_path, rvc_out, humanized_out,
                                                             do_humanize, book_dir, humanize_func)
                    if not result_path or not os.path.exists(result_path):
                        log_message(f"ФАТАЛЬНО: ошибка обработки блока {chunk_name}", book_dir)
                        sys.exit(1)

                    state['processed_chunks'].append(chunk_name)
                    save_state(book_dir, state)
                    processed_chunks.add(chunk_name)
                    log_message(f"Воркер: блок {chunk_name} успешно обработан", book_dir)

                finally:
                    rvc_busy.clear()
                    log_message(f"Воркер: RVC освобождён после блока {chunk_name}", book_dir)

                rvc_queue.task_done()

            except asyncio.TimeoutError:
                continue

    worker_task = asyncio.create_task(rvc_worker())
    semaphore = asyncio.Semaphore(edge_parallel)

    idx = next_frag
    while idx <= total_fragments:
        end_idx = min(idx + edge_parallel - 1, total_fragments)
        log_message(f"Запуск генерации фрагментов {idx}-{end_idx}", book_dir)
        tasks = []
        for i in range(idx, end_idx + 1):
            if str(i) in all_processed:
                f_p = os.path.join(tts_out, f"part_{i:04d}.wav")
                if os.path.exists(f_p) and os.path.getsize(f_p) > 1000:
                    dur = librosa.get_duration(path=f_p)
                    log_message(f"Фрагмент {i} уже существует, дл. {dur:.1f} сек", book_dir)
                    tasks.append(asyncio.sleep(0, result={'index': i, 'path': f_p, 'duration': dur, 'success': True}))
                else:
                    tasks.append(generate_one_fragment(
                        groups[i - 1], i, tts_out, semaphore, book_dir,
                        edge_speaker, sample_rate,
                        edge_delay_min, edge_delay_max, log_message))
            else:
                tasks.append(generate_one_fragment(
                    groups[i - 1], i, tts_out, semaphore, book_dir,
                    edge_speaker, sample_rate,
                    edge_delay_min, edge_delay_max, log_message))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results_by_idx = {}
        for res in results:
            if isinstance(res, Exception):
                log_message(f"Критическая ошибка в задаче: {res}", book_dir)
                sys.exit(1)
            if res and res['success']:
                results_by_idx[res['index']] = res

        for i in range(idx, end_idx + 1):
            if i not in results_by_idx:
                log_message(f"ФАТАЛЬНО: фрагмент {i} отсутствует в результатах", book_dir)
                sys.exit(1)
            res = results_by_idx[i]
            if str(i) not in state['processed_fragments']:
                state['processed_fragments'].append(str(i))
                save_state(book_dir, state)
            current_batch_indices.append(i)
            current_batch_dur += res['duration']
            state['partial_batch_indices'] = current_batch_indices
            state['partial_batch_dur'] = current_batch_dur
            save_state(book_dir, state)
            log_message(
                f"Фрагмент {i} добавлен в блок, длительность блока {current_batch_dur:.1f} сек, "
                f"диапазон: {current_batch_indices[0]}-{current_batch_indices[-1]}", book_dir)
            if current_batch_dur >= target_chunk_sec:
                indices_to_process = current_batch_indices.copy()
                dur_to_process = current_batch_dur
                chunk_name = f"chunk_{chunk_counter:04d}.wav"
                chunk_path = os.path.join(edge_chunks_dir, chunk_name)
                chunk_counter += 1
                file_list = [os.path.join(tts_out, f"part_{i:04d}.wav") for i in indices_to_process]
                await rvc_queue.put((chunk_name, chunk_path, file_list, dur_to_process, indices_to_process))
                current_batch_indices = []
                current_batch_dur = 0.0
                state['partial_batch_indices'] = []
                state['partial_batch_dur'] = 0.0
                save_state(book_dir, state)

        idx = end_idx + 1

    if current_batch_indices:
        chunk_name = f"chunk_{chunk_counter:04d}.wav"
        chunk_path = os.path.join(edge_chunks_dir, chunk_name)
        file_list = [os.path.join(tts_out, f"part_{i:04d}.wav") for i in current_batch_indices]
        await rvc_queue.put((chunk_name, chunk_path, file_list, current_batch_dur, current_batch_indices))

    await rvc_queue.join()
    stop_worker = True
    await worker_task

    # ------------------------------------------------------------
    # БЕЗУСЛОВНАЯ СКЛЕЙКА ВСЕХ НАЙДЕННЫХ ФАЙЛОВ ПО 2 ЧАСА
    # ------------------------------------------------------------
    existing_mp3s = glob.glob(os.path.join(final_out, "*.mp3"))
    if existing_mp3s:
        log_message(f"Папка final_hours уже содержит MP3-файлы ({len(existing_mp3s)} шт.). Сборка пропущена.", book_dir)
        print(f"Обнаружены готовые файлы: {existing_mp3s}")
    else:
        log_message("Начало сканирования папок для сборки книги...", book_dir)
        source_dir = humanized_out if do_humanize else rvc_out
        
        all_files = glob.glob(os.path.join(source_dir, "chunk_*.wav"))
        
        if do_humanize:
            rvc_files = glob.glob(os.path.join(rvc_out, "chunk_*.wav"))
            for rf in rvc_files:
                name = os.path.basename(rf)
                h_path = os.path.join(humanized_out, name)
                if not os.path.exists(h_path) and rf not in all_files:
                    all_files.append(rf)

        def sort_key(path):
            match = re.search(r'chunk_(\d+)\.wav', path)
            return int(match.group(1)) if match else 0
        all_files.sort(key=sort_key)

        if not all_files:
            log_message("!!! ОШИБКА СБОРКИ: не обнаружено фрагментов для склейки!", book_dir)
        else:
            log_message(f"Всего найдено {len(all_files)} фрагментов для объединения.", book_dir)
            
            TARGET_DURATION_SEC = 2 * 3600
            groups_to_merge = []
            current_group = []
            current_group_duration = 0.0

            for fp in all_files:
                try:
                    dur = librosa.get_duration(path=fp)
                except Exception:
                    dur = 600.0
                
                if current_group_duration + dur > TARGET_DURATION_SEC and current_group:
                    groups_to_merge.append(current_group)
                    current_group = [fp]
                    current_group_duration = dur
                else:
                    current_group.append(fp)
                    current_group_duration += dur
            
            if current_group:
                groups_to_merge.append(current_group)

            log_message(f"Фрагменты успешно распределены на {len(groups_to_merge)} частей (длительностью до 2 часов каждая)", book_dir)

            for f_idx, batch in enumerate(groups_to_merge, 1):
                hour_name = f"{os.path.basename(book_dir)}_part_{f_idx:02d}.mp3"
                hour_path = os.path.join(final_out, hour_name)
                
                log_message(f"Сборка итогового файла {hour_name} из {len(batch)} блоков...", book_dir)
                if len(batch) > 1:
                    success = merge_to_mp3(batch, hour_path)
                else:
                    cmd = ["ffmpeg", "-y", "-i", batch[0], "-c:a", "libmp3lame", "-b:a", "320k", hour_path]
                    success = (subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)

                if success and os.path.exists(hour_path):
                    log_message(f"Файл {hour_name} успешно создан", book_dir)
                else:
                    log_message(f"!!! Не удалось собрать файл {hour_name}", book_dir)

    log_message("Конвейер Edge TTS полностью завершён", book_dir)