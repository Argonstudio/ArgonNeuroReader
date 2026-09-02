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

# pipeline/common.py
import os
import json
import shutil
import subprocess
import librosa
import torch
import asyncio
from datetime import datetime

from pipeline.config import (SAMPLE_RATE, RVC_PYTHON, RVC_CLI, MODEL_NAME,
                            INDEX_PATH, DEFAULT_INDEX_RATE, IS_HALF, RVC_DIR,
                            VRAM_FREE_RATIO, VRAM_CHECK_INTERVAL)

# --- Логирование и состояние ---
def log_message(msg, book_dir):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    if book_dir:
        log_file = os.path.join(book_dir, 'process.log')
        try:
            with open(log_file, 'a', encoding='utf-8') as lf:
                lf.write(full_msg + '\n')
        except:
            pass

def load_state(book_dir):
    state_path = os.path.join(book_dir, 'processed_state.json')
    backup_path = state_path + '.bak'
    try:
        if os.path.exists(state_path):
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                if isinstance(state, dict) and 'processed_fragments' in state and 'processed_chunks' in state:
                    return state
    except:
        pass
    try:
        if os.path.exists(backup_path):
            with open(backup_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                log_message("Восстановлено состояние из резервной копии", book_dir)
                return state
    except:
        pass
    log_message("Файл состояния не найден или повреждён, начинаем с нуля", book_dir)
    return {'processed_fragments': [], 'processed_chunks': []}

def save_state(book_dir, state):
    state_path = os.path.join(book_dir, 'processed_state.json')
    backup_path = state_path + '.bak'
    if os.path.exists(state_path):
        try:
            shutil.copy2(state_path, backup_path)
        except:
            pass
    temp_path = state_path + '.tmp'
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, state_path)
    except Exception as e:
        log_message(f"Ошибка сохранения состояния: {e}", book_dir)

# --- Аудио утилиты ---
def merge_wav_files(file_list, output_path, normalize=True):
    if not file_list:
        return False
    list_file = os.path.join(os.path.dirname(output_path),
                             f"_concat_{os.path.basename(output_path)}.txt")
    norm_dir = os.path.join(os.path.dirname(output_path), "_norm")
    os.makedirs(norm_dir, exist_ok=True)
    normalized_files = []
    try:
        for fp in file_list:
            if not os.path.exists(fp) or os.path.getsize(fp) < 1000:
                print(f"[WARN] Пропущен битый фрагмент: {fp}")
                continue
            if normalize:
                tmp_name = f"_norm_{os.path.basename(fp)}"
                tmp_path = os.path.join(norm_dir, tmp_name)
                cmd = ["ffmpeg", "-y", "-i", fp,
                       "-ar", str(SAMPLE_RATE), "-ac", "1",
                       "-c:a", "pcm_s16le", tmp_path]
                subprocess.run(cmd, capture_output=True)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
                    normalized_files.append(tmp_path)
                else:
                    normalized_files.append(fp)
            else:
                normalized_files.append(fp)
        with open(list_file, "w", encoding="utf-8") as f:
            for fp in normalized_files:
                f.write(f"file '{os.path.abspath(fp)}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
               "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), output_path]
        subprocess.run(cmd, capture_output=True)
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)
        for fp in normalized_files:
            if fp.startswith(norm_dir) and os.path.exists(fp):
                try:
                    os.remove(fp)
                except:
                    pass
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        try:
            dur = librosa.get_duration(path=output_path)
            if dur < 0.5:
                os.remove(output_path)
                return False
        except:
            pass
        return True
    return False

def merge_to_mp3(file_list, output_path):
    if not file_list:
        return False
    list_path = f"_concat_mp3_{os.path.basename(output_path)}.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for fp in file_list:
                if not os.path.exists(fp) or os.path.getsize(fp) < 1000:
                    continue
                f.write(f"file '{os.path.abspath(fp)}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
               "-c:a", "libmp3lame", "-b:a", "320k", output_path]
        subprocess.run(cmd, capture_output=True)
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return os.path.exists(output_path)

# --- VRAM и RVC ---
async def wait_for_free_vram(book_dir, required_ratio=VRAM_FREE_RATIO, check_interval=VRAM_CHECK_INTERVAL):
    if not torch.cuda.is_available():
        log_message("CUDA не доступна, пропускаем проверку памяти", book_dir)
        return True
    while True:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        free_ratio = free / total
        if free_ratio >= required_ratio:
            log_message(f"Видеопамять: свободно {free/(1024**3):.1f} ГБ из {total/(1024**3):.1f} ({free_ratio*100:.1f}%) – OK", book_dir)
            return True
        else:
            log_message(f"Видеопамять: занято {(total-free)/(1024**3):.1f} ГБ ({free_ratio*100:.1f}% свободно). Ожидание {check_interval//60} мин...", book_dir)
            await asyncio.sleep(check_interval)

def clean_edge_audio(input_path: str, output_path: str = None) -> str:
    """
    Очищает аудио от артефактов Edge-TTS (микропаузы, щелчки, скачки тона).
    Безопасно заменяет цифровой ноль на микрошум для защиты RVC от зависаний.
    """
    import os
    import soundfile as sf
    import numpy as np

    if output_path is None:
        output_path = input_path + ".cleaned.wav"
        replace_original = True
    else:
        replace_original = False

    # 1. Читаем аудиофайл
    data, sr = sf.read(input_path)
    
    # Если аудио пустое, возвращаем как есть
    if data.size == 0:
        return input_path

    # Конвертируем стерео в моно, если это необходимо для RVC
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # 2. БЕЗОПАСНАЯ ОБРЕЗКА ТИШИНЫ (только явные нули в самом начале и конце)
    # Ищем все элементы, которые физически не равны нулю
    nonzero_indices = np.nonzero(data)[0]
    
    if nonzero_indices.size > 0:
        start_idx = nonzero_indices[0]
        end_idx = nonzero_indices[-1]
        # Обрезаем только если есть что обрезать по краям
        data = data[start_idx:end_idx + 1]
    
    # 3. УДАЛЕНИЕ ЦИФРОВОГО НУЛЯ (Внутри самого файла)
    # Находим абсолютные нули на стыках слов, из-за которых виснет Harvest/Crepe
    mask = data == 0.0
    if np.any(mask):
        # Генерируем тишайший, неслышимый белый шум (-90 дБ)
        data[mask] = np.random.normal(0, 1e-5, size=np.sum(mask))
    
    # 4. МЯГКОЕ ОГРАНИЧЕНИЕ ПИКОВИ ХАРАКТЕРИСТИК (Soft Limiter)
    threshold = 0.9
    data = np.clip(data, -threshold, threshold)
    
    # 5. НОРМАЛИЗАЦИЯ ЗВУКА
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    
    # 6. ЗАПИСЬ В ФАЙЛ
    sf.write(output_path, data, sr)
    
    if replace_original:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            os.rename(output_path, input_path)
        except OSError:
            import time
            time.sleep(0.5)
            if os.path.exists(input_path):
                os.remove(input_path)
            os.rename(output_path, input_path)
        return input_path
        
    return output_path


def process_rvc_block(chunk_path, rvc_out, humanized_out, do_humanize, book_dir, humanize_func):
    fname = os.path.basename(chunk_path)
    out_p = os.path.join(rvc_out, fname)
    if os.path.exists(out_p) and os.path.getsize(out_p) > 1000:
        log_message(f"RVC для {fname} уже обработан, пропускаем", book_dir)
    else:
        cmd = [RVC_PYTHON, RVC_CLI,
               "--f0up_key", "0",
               "--input_path", chunk_path,
               "--model_name", MODEL_NAME,
               "--index_path", INDEX_PATH,
               "--f0method", "rmvpe",
               "--opt_path", out_p,
               "--index_rate", str(DEFAULT_INDEX_RATE),
               "--device", "cuda:0",
               "--is_half", IS_HALF,
               "--filter_radius", "1",
               "--rms_mix_rate", "0.15",
               "--protect", "0.25",
               "--resample", "0"]
        subprocess.run(cmd, cwd=RVC_DIR)
        if not os.path.exists(out_p) or os.path.getsize(out_p) < 1000:
            log_message(f"Ошибка RVC для {fname}: выходной файл не создан или слишком мал", book_dir)
            return None
        log_message(f"RVC завершён для {fname}", book_dir)

    if do_humanize:
        f_out = os.path.join(humanized_out, fname)
        if os.path.exists(f_out) and os.path.getsize(f_out) > 1000:
            log_message(f"Humanize для {fname} уже обработан, пропускаем", book_dir)
            return f_out
        else:
            try:
                humanize_func(out_p, f_out)
                log_message(f"Humanize завершён для {fname}", book_dir)
                return f_out
            except Exception as e:
                # ВАЖНОЕ ИЗМЕНЕНИЕ: вместо потери блока используем RVC-файл напрямую
                log_message(f"Ошибка Humanize для {fname}: {e}. Используется RVC без humanize.", book_dir)
                return out_p   # fallback, чтобы блок не пропадал
    else:
        return out_p