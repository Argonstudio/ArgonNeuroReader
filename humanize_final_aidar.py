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

import soundfile as sf
import numpy as np
from scipy import signal
import random

# --- 1. Инструменты чистки (оставлены как есть) ---
def apply_lowpass(audio, sr, cutoff_hz=4500, steepness=10, mix=1.0):
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist or mix == 0: return audio
    normal_cut = cutoff_hz / nyquist
    b, a = signal.butter(steepness, normal_cut, btype='low')
    filtered = signal.filtfilt(b, a, audio)
    return filtered * mix + audio * (1 - mix)

def apply_highpass(audio, sr, cutoff_hz=50, steepness=6):
    nyquist = sr / 2.0
    normal_cut = cutoff_hz / nyquist
    b, a = signal.butter(steepness, normal_cut, btype='high')
    return signal.filtfilt(b, a, audio)

def notch_reject(audio, sr, center_freq=10000, q=8.0, gain_db=-20):
    nyquist = sr / 2.0
    if center_freq >= nyquist: return audio
    bw = center_freq / q
    low = max(10, center_freq - bw/2)
    high = min(nyquist-10, center_freq + bw/2)
    b, a = signal.butter(4, [low, high], btype='band', fs=sr)
    filtered = signal.filtfilt(b, a, audio)
    gain_lin = 10 ** (gain_db / 20)
    return audio + (gain_lin - 1) * filtered

def soft_compressor(audio, sr, threshold_db=-20, ratio=2.0):
    threshold_lin = 10 ** (threshold_db / 20)
    envelope = np.abs(audio)
    gain = np.ones_like(audio)
    mask = envelope > threshold_lin
    if np.any(mask):
        gain[mask] = (threshold_lin / envelope[mask]) ** (1 - 1/ratio)
    gain = np.clip(gain, 0.5, 1.0)
    b, a = signal.butter(2, 20/(sr/2), btype='low')
    gain = signal.filtfilt(b, a, gain)
    return audio * gain

# --- 2. НОВЫЕ ИНСТРУМЕНТЫ ДЛЯ "ОЧЕЛОВЕЧИВАНИЯ" ---

# 2.1 Эквалайзер (с подъёмом низов и верхов)
def eq_smile_curve(audio, sr):
    """Применяет 'улыбку' для добавления тепла и воздуха."""
    # 1. Подъём низких частот (глубина)
    b_low, a_low = signal.butter(2, 120/(sr/2), btype='low')
    low_signal = signal.filtfilt(b_low, a_low, audio)
    gain_low = 1.6  # +4 дБ
    audio = audio + (gain_low - 1) * low_signal

    # 2. Подъём высоких частот (воздух)
    b_high, a_high = signal.butter(2, 7000/(sr/2), btype='high')
    high_signal = signal.filtfilt(b_high, a_high, audio)
    gain_high = 1.4  # +3 дБ
    audio = audio + (gain_high - 1) * high_signal
    return audio

# 2.2 Эксайтер (гармоническое обогащение)
def gentle_exciter(audio, sr, drive=0.3, mix=0.4):
    """Добавляет гармоники для яркости и 'дороговизны' звука."""
    excited = np.tanh(audio * (1 + drive))
    return audio * (1 - mix) + excited * mix

# 2.3 Пространство (комнатная реверберация)
def add_room(audio, sr, room_size=0.15, wet=0.15, dry=0.85):
    """Имитирует небольшое пространство для объёма."""
    delay_samples = int(sr * 0.02)  # 20 мс
    if delay_samples >= len(audio): return audio
    early_reflections = np.zeros_like(audio)
    early_reflections[delay_samples:] = audio[:-delay_samples] * 0.6
    
    # Добавляем второе, более тихое отражение
    second_delay = int(sr * 0.035) # 35 мс
    early_reflections[second_delay:] += audio[:-second_delay] * 0.3
    
    wet_signal = early_reflections
    return audio * dry + wet_signal * wet

# 2.4 Случайная микро-модуляция высоты тона (Pitch Jitter)
def apply_pitch_jitter(audio, sr, depth_factor=0.5):
    """Добавляет естественные микро-колебания высоты тона."""
    t = np.arange(len(audio)) / sr
    # Используем комбинацию двух синусоид для более естественного ощущения
    jitter = depth_factor * 1e-4 * (np.sin(2 * np.pi * 0.9 * t) + 0.6 * np.sin(2 * np.pi * 1.3 * t))
    original_indices = np.arange(len(audio))
    new_indices = original_indices + jitter * sr
    new_indices = np.clip(new_indices, 0, len(audio) - 1)
    # Линейная интерполяция для нового сдвига
    from scipy.interpolate import interp1d
    interpolator = interp1d(original_indices, audio, kind='linear', fill_value='extrapolate')
    return interpolator(new_indices)

# --- 3. ГЛАВНАЯ ФУНКЦИЯ: сводим всё вместе ---
def humanize_audio_aidar(input_path, output_path):
    data, sr = sf.read(input_path)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    data = data / (np.abs(data).max() + 1e-6) * 0.95

    # Шаг 1: Бережная чистка (без фанатизма)
    data = apply_highpass(data, sr, cutoff_hz=50, steepness=6)
    data = apply_lowpass(data, sr, cutoff_hz=6500, steepness=8, mix=1.0) # Поднял частоту среза 
    data = notch_reject(data, sr, center_freq=10000, q=8.0, gain_db=-20)
    
    # ---------------------------------------------------------
    # Шаг 2: Креативная обработка для "глубины" и "красоты"
    # ---------------------------------------------------------
    # 1. Эквалайзер (Тональный баланс)
    data = eq_smile_curve(data, sr)
    
    # 2. Эксайтер (Гармоническое обогащение)
    data = gentle_exciter(data, sr, drive=0.25, mix=0.3)
    
    # 3. Пространство (Объём)
    data = add_room(data, sr, room_size=0.1, wet=0.12, dry=0.88)
    
    # 4. Микро-модуляция (Естественность)
    data = apply_pitch_jitter(data, sr, depth_factor=0.6)

    # Шаг 3: Финальный компрессор для "склейки" (делает звук более плотным и профессиональным)
    data = soft_compressor(data, sr, threshold_db=-18, ratio=1.8)

    # Финальная нормализация
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.85

    sf.write(output_path, data, sr)

# Оставляем псевдоним для совместимости
humanize_audio = humanize_audio_aidar



