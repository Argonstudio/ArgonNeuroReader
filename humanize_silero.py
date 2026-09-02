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

import cupy as cp
from cupyx.scipy import signal as cp_signal
import numpy as np
import soundfile as sf
import os

def remove_clicks_gpu(audio_gpu, sr):
    nyq = sr / 2.0
    b_hp, a_hp = cp_signal.butter(4, 2000/nyq, btype='high')
    audio_hp = cp_signal.filtfilt(b_hp, a_hp, audio_gpu)
    frame_len = int(sr * 0.001)
    win = cp.ones(frame_len, dtype=cp.float32) / frame_len
    envelope = cp.sqrt(cp.convolve(audio_hp**2, win, mode='same'))
    mask = cp.abs(audio_hp) > 5.0 * (envelope + 1e-12)
    mask_np = cp.asnumpy(mask)
    if not np.any(mask_np):
        return audio_gpu
    cleaned = audio_gpu.copy()
    indices = np.where(mask_np)[0]
    for i in indices:
        left = max(0, i-10)
        right = min(len(audio_gpu)-1, i+10)
        if right > left:
            y_left = cp.asnumpy(audio_gpu[left])
            y_right = cp.asnumpy(audio_gpu[right])
            val = np.interp(i, [left, right], [y_left, y_right])
            cleaned[i] = cp.asarray(val)
    return cleaned

def eq_metal_gpu(audio_gpu, sr):
    nyquist = sr / 2.0
    bands = [(4200, 4.0, -9), (5100, 5.0, -15), (5600, 6.0, -12), (6200, 5.0, -12), (7500, 5.0, -7)]
    for freq, q, gain_db in bands:
        gain_lin = 10 ** (gain_db / 20)
        bw = freq / q
        low = max(10, freq - bw/2)
        high = min(nyquist-10, freq + bw/2)
        if low >= high: continue
        b, a = cp_signal.butter(4, [low, high], btype='band', fs=sr)
        filtered = cp_signal.filtfilt(b, a, audio_gpu)
        audio_gpu = audio_gpu + (gain_lin - 1) * filtered
    return audio_gpu

def notch_filter_gpu(audio_gpu, sr, freq, q, gain_db):
    nyquist = sr / 2.0
    if freq >= nyquist: return audio_gpu
    bw = freq / q
    low = max(10, freq - bw/2)
    high = min(nyquist-10, freq + bw/2)
    b, a = cp_signal.butter(4, [low, high], btype='band', fs=sr)
    filtered = cp_signal.filtfilt(b, a, audio_gpu)
    gain_lin = 10 ** (gain_db / 20)
    return audio_gpu + (gain_lin - 1) * filtered

def lowpass_gentle_gpu(audio_gpu, sr, cutoff=6500, steepness=4, mix=0.8):
    nyquist = sr / 2.0
    norm_cut = min(cutoff, nyquist-100) / nyquist
    b, a = cp_signal.butter(steepness, norm_cut, btype='low')
    filtered = cp_signal.filtfilt(b, a, audio_gpu)
    return filtered * mix + audio_gpu * (1 - mix)

def simple_gate_gpu(audio_gpu, sr, threshold_db=-48):
    frame_len = int(sr * 0.02)
    win = cp.ones(frame_len, dtype=cp.float32) / frame_len
    envelope = cp.sqrt(cp.convolve(audio_gpu**2, win, mode='same'))
    gain = cp.ones_like(audio_gpu)
    gain[envelope < 10**(threshold_db/20)] = 0.05
    b, a = cp_signal.butter(2, 20/(sr/2), btype='low')
    gain = cp_signal.filtfilt(b, a, gain)
    return audio_gpu * gain

def humanize_silero(input_path, output_path):
    """
    Основная функция: с жесткой очисткой памяти GPU.
    """
    # 1. ОЧИСТКА ПЕРЕД РАБОТОЙ
    cp.get_default_memory_pool().free_all_blocks()
    
    if not os.path.exists(input_path):
        return

    data_np, sr = sf.read(input_path)
    if data_np.ndim == 2:
        data_np = np.mean(data_np, axis=1)
    
    data_np = data_np.astype(np.float32)
    max_val = np.abs(data_np).max()
    if max_val > 0:
        data_np = data_np / max_val * 0.95

    # 2. ПЕРЕНОС НА GPU
    data = cp.asarray(data_np)

    # 3. ФИЛЬТРЫ
    data = remove_clicks_gpu(data, sr)
    data = eq_metal_gpu(data, sr)
    data = notch_filter_gpu(data, sr, freq=5200, q=12.0, gain_db=-10)
    data = notch_filter_gpu(data, sr, freq=5700, q=10.0, gain_db=-6)
    data = lowpass_gentle_gpu(data, sr, cutoff=6500, steepness=4, mix=0.8)
    data = simple_gate_gpu(data, sr, threshold_db=-48)

    # 4. ВЫГРУЗКА И ОЧИСТКА
    peak = cp.abs(data).max()
    if peak > 0:
        data = data / peak * 0.707
        
    final_data_np = cp.asnumpy(data)
    sf.write(output_path, final_data_np, sr)

    # 5. ФИНАЛЬНОЕ ОСВОБОЖДЕНИЕ ПАМЯТИ
    del data
    cp.get_default_memory_pool().free_all_blocks()

