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


# pipeline/silero_tts.py
import os
import torch
import numpy as np
import soundfile as sf
import librosa
import subprocess
import time
import gc
import glob
import traceback

# Попытка импорта параметров финальной сборки из конфига
try:
    from pipeline.config import FINAL_MP3_DURATION_SEC, TARGET_CHUNK_SEC
except ImportError:
    FINAL_MP3_DURATION_SEC = 3600      # 1 час по умолчанию
    TARGET_CHUNK_SEC = 600             # 10 минут, приблизительная длительность одного блока Silero

# Константа размера фрагмента для Silero
SILERO_CHUNK_SYMBOLS = 25000

def prepare_groups_silero(paragraphs, chunk_limit):
    """
    Группирует параграфы в текстовые фрагменты так,
    чтобы каждый фрагмент не превышал chunk_limit символов.
    """
    groups = []
    curr, c_len = [], 0
    for p in paragraphs:
        if c_len + len(p) > chunk_limit and curr:
            groups.append(" ".join(curr))
            curr = [p]
            c_len = len(p)
        else:
            curr.append(p)
            c_len += len(p)
    if curr:
        groups.append(" ".join(curr))
    return groups

def voice_silero(text: str, path: str, model, speaker: str, stretch_rate: float,
                 sample_rate: int, humanize_silero_func) -> bool:
    try:
        start_time = time.perf_counter()
        sub_chunks = [text[i:i + 800] for i in range(0, len(text), 800)]
        audio_data = []
        speaker_name = speaker if speaker in ("aidar", "eugene") else "aidar"
        
        with torch.inference_mode():
            for chunk in sub_chunks:
                clean_chunk = str(chunk).strip()
                if not clean_chunk:
                    continue
                audio_tensor = model.apply_tts(
                    text=clean_chunk,
                    speaker=speaker_name,
                    sample_rate=48000
                )
                audio_chunk = audio_tensor.detach().cpu().numpy().flatten()
                audio_data.append(audio_chunk)
                del audio_tensor
        
        if not audio_data:
            return False
        
        full_y = np.concatenate(audio_data).astype(np.float32)
        
        if stretch_rate != 1.0:
            full_y = librosa.effects.time_stretch(full_y, rate=stretch_rate)
        
        if sample_rate != 48000:
            full_y = librosa.resample(full_y, orig_sr=48000, target_sr=sample_rate)
        
        temp_path = path + ".temp.wav"
        sf.write(temp_path, full_y, sample_rate)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            humanize_silero_func(temp_path, path)
        except Exception as e:
            print(f" [Humanize Error] {e}")
            if os.path.exists(temp_path):
                import shutil
                shutil.copy(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ Готово (Silero v5) за {elapsed:.1f} сек")
        return True
        
    except Exception as e:
        print(f" [Silero v5] Ошибка: {e}")
        traceback.print_exc()
        return False

def process_silero(book_dir, groups, tts_out, rvc_out, humanized_out, final_out,
                   do_humanize, silero_speaker, stretch_rate,
                   rvc_python, rvc_cli, model_name, index_path, default_index_rate,
                   is_half, humanize_func, humanize_silero_func, merge_to_mp3, log_message,
                   sample_rate=48000):
    """
    Запускает синтез Silero для уже готовых текстовых групп,
    затем RVC, humanize и склейку в MP3.
    """
    
    total_groups = len(groups)
    log_message(f"Silero v5: получено {total_groups} фрагментов", book_dir)
    
    # Сохраняем итоговый текст для отладки
    full_text = "\n\n".join(groups)
    text_file_path = os.path.join(book_dir, "silero_text.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    log_message(f"Текст для Silero сохранён в {text_file_path}", book_dir)
    
    # --- Синтез Silero ---
    silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language='ru',
                                     speaker='v5_ru',
                                     trust_repo=True)
    silero_model.to('cuda')
    
    silero_success = 0
    for idx, text in enumerate(groups, 1):
        f_p = os.path.join(tts_out, f"part_{idx:04d}.wav")
        if os.path.exists(f_p) and os.path.getsize(f_p) > 1000:
            log_message(f"Silero фрагмент {idx}/{total_groups} уже существует, пропускаем", book_dir)
            silero_success += 1
            continue
        
        log_message(f"Генерация Silero фрагмента {idx}/{total_groups}", book_dir)
        if voice_silero(text, f_p, silero_model, silero_speaker, stretch_rate,
                        sample_rate, humanize_silero_func):
            silero_success += 1
            log_message(f"Silero фрагмент {idx} готов", book_dir)
        else:
            log_message(f"Ошибка Silero фрагмента {idx}", book_dir)
    
    del silero_model
    gc.collect()
    torch.cuda.empty_cache()
    
    log_message(f"Silero: успешно {silero_success} из {total_groups}", book_dir)
    
    # --- ШАГ 1: RVC для всех имеющихся WAV (ЯВНЫЙ СПИСОК вместо glob) ---
    rvc_processed = []
    for idx in range(1, total_groups + 1):
        fname = f"part_{idx:04d}.wav"
        silero_path = os.path.join(tts_out, fname)
        rvc_path = os.path.join(rvc_out, fname)
        
        # Проверяем, что Silero-файл существует
        if not os.path.exists(silero_path) or os.path.getsize(silero_path) < 1000:
            log_message(f"⚠️ Silero-файл {fname} отсутствует или битый, пропускаем", book_dir)
            continue
        
        # Запускаем RVC если нужно
        if not os.path.exists(rvc_path) or os.path.getsize(rvc_path) < 1000:
            log_message(f"Запуск RVC для {fname}", book_dir)
            cmd = [rvc_python, rvc_cli,
                   "--f0up_key", "0",
                   "--input_path", silero_path,
                   "--model_name", model_name,
                   "--index_path", index_path,
                   "--f0method", "rmvpe",
                   "--opt_path", rvc_path,
                   "--index_rate", str(default_index_rate),
                   "--device", "cuda:0",
                   "--is_half", is_half,
                   "--filter_radius", "1",
                   "--rms_mix_rate", "0.15",
                   "--protect", "0.25",
                   "--resample", "0"]
            rvc_dir = os.path.dirname(os.path.dirname(rvc_cli))
            subprocess.run(cmd, cwd=rvc_dir)
            
            if not os.path.exists(rvc_path) or os.path.getsize(rvc_path) < 1000:
                log_message(f"Ошибка RVC для {fname}: выходной файл не создан", book_dir)
                continue
            else:
                log_message(f"RVC завершён для {fname}", book_dir)
        else:
            log_message(f"RVC для {fname} уже существует, пропускаем", book_dir)
        
        rvc_processed.append(rvc_path)
    
    log_message(f"RVC: успешно обработано {len(rvc_processed)} из {total_groups} файлов", book_dir)
    
    # --- ШАГ 2: Humanize для всех успешных RVC-файлов (если нужно) ---
    if do_humanize:
        final_processed = []
        for rvc_path in rvc_processed:
            fname = os.path.basename(rvc_path)
            f_out = os.path.join(humanized_out, fname)
            if not os.path.exists(f_out) or os.path.getsize(f_out) < 1000:
                log_message(f"Humanize для {fname}", book_dir)
                try:
                    humanize_func(rvc_path, f_out)
                    final_processed.append(f_out)
                    log_message(f"Humanize завершён для {fname}", book_dir)
                except Exception as e:
                    log_message(f"Ошибка Humanize для {fname}: {e}. Используется RVC без humanize.", book_dir)
                    final_processed.append(rvc_path)   # fallback
            else:
                log_message(f"Humanize для {fname} уже существует", book_dir)
                final_processed.append(f_out)
    else:
        final_processed = rvc_processed
    
    log_message(f"Humanize: успешно обработано {len(final_processed)} файлов", book_dir)
    
    if not final_processed:
        log_message("Нет успешно обработанных файлов для склейки", book_dir)
        return
    
    # --- Динамическая сборка в MP3 нужной длительности ---
    blocks_per_file = max(1, FINAL_MP3_DURATION_SEC // TARGET_CHUNK_SEC)
    log_message(
        f"Сборка итоговых MP3 (Silero): {len(final_processed)} файлов, "
        f"по {blocks_per_file} в группе (~{TARGET_CHUNK_SEC * blocks_per_file // 60} мин)",
        book_dir
    )
    
    for i in range(0, len(final_processed), blocks_per_file):
        batch = final_processed[i:i + blocks_per_file]
        hour_name = f"{os.path.basename(book_dir)}_hour_{i // blocks_per_file + 1:02d}.mp3"
        hour_path = os.path.join(final_out, hour_name)
        
        if len(batch) > 1:
            success = merge_to_mp3(batch, hour_path)
        else:
            cmd = ["ffmpeg", "-y", "-i", batch[0], "-c:a", "libmp3lame", "-b:a", "320k", hour_path]
            success = (subprocess.run(cmd, capture_output=True).returncode == 0)
        
        if success and os.path.exists(hour_path):
            log_message(f"Создан файл {hour_name} из {len(batch)} блоков", book_dir)
        else:
            log_message(f"Ошибка создания {hour_name}", book_dir)
    
    log_message("Конвейер Silero TTS полностью завершён", book_dir)
