import numpy as np
import soundfile as sf
from scipy import signal

# ----------------------------------------------------------------------
# 1. Удаление высокочастотных кликов (треск)
# ----------------------------------------------------------------------
def remove_clicks(audio, sr, threshold_factor=6.0, click_duration_ms=2.0):
    """
    Удаляет короткие щелчки и выбросы (клики) методом обнаружения по
    высокочастотной составляющей и последующей интерполяцией.
    """
    nyq = sr / 2.0
    highpass_cutoff = 2000
    b_hp, a_hp = signal.butter(4, highpass_cutoff/nyq, btype='high')
    audio_hp = signal.filtfilt(b_hp, a_hp, audio)

    frame_len = int(sr * 0.001)
    envelope = np.sqrt(np.convolve(audio_hp**2, np.ones(frame_len)/frame_len, mode='same'))

    mask = np.abs(audio_hp) > threshold_factor * (envelope + 1e-12)

    clicks = []
    start = None
    for i, val in enumerate(mask):
        if val:
            if start is None:
                start = i
        else:
            if start is not None:
                clicks.append((start, i-1))
                start = None
    if start is not None:
        clicks.append((start, len(mask)-1))

    if not clicks:
        return audio

    win_samples = int(sr * click_duration_ms / 1000.0)
    cleaned = audio.copy()

    for cl_start, cl_end in clicks:
        left = max(0, cl_start - win_samples)
        right = min(len(audio) - 1, cl_end + win_samples)
        if right - left < 2:
            continue
        x_left, x_right = left, right
        y_left, y_right = cleaned[left], cleaned[right]
        for idx in range(left, right+1):
            frac = (idx - x_left) / (x_right - x_left)
            cleaned[idx] = y_left * (1 - frac) + y_right * frac

    for cl_start, cl_end in clicks:
        smooth_radius = int(sr * 0.0005)
        region_start = max(0, cl_start - smooth_radius)
        region_end = min(len(cleaned)-1, cl_end + smooth_radius)
        if region_end - region_start <= 2:
            continue
        segment = cleaned[region_start:region_end+1]
        segment = signal.medfilt(segment, kernel_size=min(5, len(segment)))
        cleaned[region_start:region_end+1] = segment

    peak = np.abs(cleaned).max()
    if peak > 0.99:
        cleaned = cleaned / peak * 0.95
    return cleaned


# ----------------------------------------------------------------------
# 2. Удаление глухих щелчков в паузах (между фразами)
# ----------------------------------------------------------------------
def remove_silence_clicks(audio, sr,
                          silence_thresh_db=-40,
                          min_silence_duration=0.2,
                          click_max_duration_ms=25,
                          click_threshold_db=12,
                          fade_ms=5):
    """
    Удаляет короткие щелчки, возникающие в тишине между фразами.
    """
    frame_len = int(sr * 0.005)
    rms = np.sqrt(np.convolve(audio**2, np.ones(frame_len)/frame_len, mode='same'))
    rms_db = 20 * np.log10(rms + 1e-12)

    silence_mask = rms_db < silence_thresh_db
    min_sil_samples = int(sr * min_silence_duration)

    padded = np.pad(silence_mask, (1, 1), constant_values=False)
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    cleaned = audio.copy()

    for start, end in zip(starts, ends):
        if end - start + 1 < min_sil_samples:
            continue

        silence_region = audio[start:end+1]
        noise_rms = np.sqrt(np.mean(silence_region**2)) + 1e-12
        threshold_linear = noise_rms * (10 ** (click_threshold_db / 20))

        peak_mask = np.abs(silence_region) > threshold_linear
        if not np.any(peak_mask):
            continue

        events = []
        ev_start = None
        for i, val in enumerate(peak_mask):
            if val:
                if ev_start is None:
                    ev_start = i
            else:
                if ev_start is not None:
                    events.append((ev_start, i-1))
                    ev_start = None
        if ev_start is not None:
            events.append((ev_start, len(peak_mask)-1))

        max_click_samples = int(sr * click_max_duration_ms / 1000.0)
        fade_samples = int(sr * fade_ms / 1000.0)

        for ev_s, ev_e in events:
            ev_length = ev_e - ev_s + 1
            if ev_length > max_click_samples:
                continue

            abs_ev_s = start + ev_s
            abs_ev_e = start + ev_e

            left = max(start, abs_ev_s - fade_samples)
            right = min(end, abs_ev_e + fade_samples)

            if right - left < 2:
                continue

            x_left, x_right = left, right
            y_left, y_right = cleaned[left], cleaned[right]
            for idx in range(left, right+1):
                frac = (idx - x_left) / (x_right - x_left)
                cleaned[idx] = y_left * (1 - frac) + y_right * frac

    peak = np.abs(cleaned).max()
    if peak > 0.99:
        cleaned = cleaned / peak * 0.95
    return cleaned


# ----------------------------------------------------------------------
# 3. Одиночный деэссер (с регулируемой добротностью Q)
# ----------------------------------------------------------------------
def deesser(audio, sr, center_freq=6000, q=2.0, threshold_db=-20, gain_reduction_db=-10):
    """
    Подавление шипящих с adjustable Q (добротностью).
    """
    nyquist = sr / 2.0
    bandwidth = center_freq / max(q, 0.5)
    low = max(10, center_freq - bandwidth/2)
    high = min(nyquist - 10, center_freq + bandwidth/2)
    if low >= high or low >= nyquist:
        return audio

    b, a = signal.butter(4, [low, high], btype='band', fs=sr)
    filtered = signal.filtfilt(b, a, audio)

    frame_len = int(sr * 0.005)
    envelope = np.convolve(filtered**2, np.ones(frame_len)/frame_len, mode='same')
    envelope_db = 10 * np.log10(envelope + 1e-12)

    gain = np.ones_like(audio)
    reduction_linear = 10 ** (gain_reduction_db / 20)
    mask = envelope_db > threshold_db
    gain[mask] = reduction_linear

    gain = signal.medfilt(gain, kernel_size=31)
    b_smooth, a_smooth = signal.butter(2, 10/(sr/2), btype='low')
    gain = signal.filtfilt(b_smooth, a_smooth, gain)

    deessed = audio - (1 - gain) * filtered
    max_val = np.abs(deessed).max()
    if max_val > 0.99:
        deessed = deessed / (max_val + 1e-6) * 0.99
    return deessed


# ----------------------------------------------------------------------
# 4. Многополосный деэссер (последовательное применение)
# ----------------------------------------------------------------------
def multiband_deesser(audio, sr, bands_config):
    """
    Применяет несколько деэссеров с разными частотами, Q, порогами и ослаблением.

    bands_config: список словарей, каждый содержит:
        - 'freq': центральная частота (Гц)
        - 'q': добротность (чем выше, тем уже полоса)
        - 'threshold_db': порог срабатывания (дБ)
        - 'reduction_db': ослабление (отрицательное число)
    """
    for band in bands_config:
        audio = deesser(audio, sr,
                        center_freq=band['freq'],
                        q=band.get('q', 2.0),
                        threshold_db=band['threshold_db'],
                        gain_reduction_db=band['reduction_db'])
    return audio


# ----------------------------------------------------------------------
# 5. Главная функция очеловечивания
# ----------------------------------------------------------------------
def humanize_audio(input_path, output_path,
                   vibrato_depth_percent=0.3,
                   tremolo_depth=0.02,
                   low_shelf_gain_db=0.5,
                   bands_config=None):
    """
    Применяет комплексную обработку голоса:
      - удаление высокочастотного треска (кликов)
      - удаление глухих щелчков в паузах
      - очень мягкое вибрато
      - лёгкое тремоло
      - подъём низких частот (low-shelf)
      - многополосный деэссер (6 полос по умолчанию) — убирает «ссс» и высокочастотный звон

    Параметры:
        bands_config: список словарей для многополосного деэссера.
                      Если None, используется предустановленная 6-полосная конфигурация.
    """
    data, sr = sf.read(input_path)
    stereo = data.ndim == 2
    if stereo:
        data = np.mean(data, axis=1)
    data = data / (np.abs(data).max() + 1e-6) * 0.95

    # -- Удаление артефактов --
    data = remove_clicks(data, sr)
    data = remove_silence_clicks(data, sr)

    # -- Вибрато --
    vibrato_freq = 4.0
    max_delay_sec = vibrato_depth_percent / 100.0 / vibrato_freq
    max_delay_samples = max(1, int(sr * max_delay_sec))
    t = np.arange(len(data)) / sr
    delay = max_delay_samples * np.sin(2 * np.pi * vibrato_freq * t)
    src_idx = np.arange(len(data)) - delay
    src_idx = np.clip(src_idx, 0, len(data)-1)
    idx0 = np.floor(src_idx).astype(int)
    idx1 = np.clip(idx0+1, 0, len(data)-1)
    frac = src_idx - idx0
    data = data[idx0] * (1 - frac) + data[idx1] * frac

    # -- Тремоло --
    tremolo_freq = 3.5
    t = np.arange(len(data)) / sr
    env = 1 + tremolo_depth * np.sin(2 * np.pi * tremolo_freq * t)
    data = data * env

    # -- Low-shelf фильтр (подъём низких) --
    if low_shelf_gain_db != 0:
        cutoff_hz = 800
        nyquist = sr / 2.0
        if cutoff_hz < nyquist:
            gain = 10**(low_shelf_gain_db/20)
            b, a = signal.butter(2, cutoff_hz/nyquist, btype='low')
            filtered = signal.filtfilt(b, a, data)
            data = data * 0.9 + filtered * 0.1 * gain

    # -- Деэссер (многополосный) --
    if bands_config is None:
        # 9-полосная конфигурация по умолчанию (проверено на мужском голосе)
        bands_config = [
            {'freq': 4500, 'q': 3.0, 'threshold_db': -24, 'reduction_db': -5},   # лёгкое касание нижней границы
            {'freq': 5200, 'q': 3.5, 'threshold_db': -26, 'reduction_db': -7},
            {'freq': 6000, 'q': 4.0, 'threshold_db': -27, 'reduction_db': -9},
            {'freq': 6800, 'q': 4.0, 'threshold_db': -28, 'reduction_db': -10},
            {'freq': 7800, 'q': 4.5, 'threshold_db': -30, 'reduction_db': -14},  # основной свист
            {'freq': 8800, 'q': 5.0, 'threshold_db': -32, 'reduction_db': -14},
            {'freq': 10000, 'q': 5.5, 'threshold_db': -34, 'reduction_db': -14},
            {'freq': 11500, 'q': 6.0, 'threshold_db': -36, 'reduction_db': -14},  # очень лёгкое
            {'freq': 12500, 'q': 6.0, 'threshold_db': -36, 'reduction_db': -14},  # очень лёгкое
        ]
    data = multiband_deesser(data, sr, bands_config)

    # Финальная нормализация -3 dBFS
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.707

    if stereo:
        data = np.column_stack((data, data))
    sf.write(output_path, data, sr)


# ----------------------------------------------------------------------
# 6. Пример использования (при запуске файла напрямую)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Если нужна своя конфигурация, передайте её в bands_config
    # humanize_audio("input.wav", "output_humanized.wav",
    #                vibrato_depth_percent=0.3,
    #                tremolo_depth=0.02,
    #                low_shelf_gain_db=0.5,
    #                bands_config=my_custom_bands)

    # Или используйте стандартную (6 полос)
    humanize_audio("input.wav", "output_humanized.wav")