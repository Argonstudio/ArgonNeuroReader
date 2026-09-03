# Ключевые возможности

**ArgonNeuroReader** это консольная Python программа для озвучки текста, генерации аудиокниг.

## 📑 Навигация / Table of Contents

| 🇷🇺 Русская версия | 🇬🇧 English Version |
| :--- | :--- |
| • [Что делает программа?](#что-делает-программа) | • [What does the program do?](#what-does-the-program-do) |
| • [Что получается в итоге?](#что-получается-в-итоге) | • [What is the final result?](#what-is-the-final-result) |
| • [Обзор возможностей конвейера](#два-режима-синтеза-речи-две-разные-нейросети) | • [Pipeline Features Overview](#two-speech-synthesis-modes-two-different-neural-networks) |
| • [🛠️ Системные требования](#system-requirements) | • [💻 System Requirements](#system-requirements) |
| • [🚀 Пошаговая установка](#шаг-1-клонирование-репозитория) | • [🚀 Step-by-Step Installation](#-step-1-cloning-the-repository) |
| • [📁 Итоговая структура папок](#-итоговая-структура-проекта-после-установки) | • [📁 Final Project Structure](#-final-project-structure-after-installation) |
| • [👤 Автор и благодарности](#автор) | • [👤 Author & Acknowledgements](#-author) |
| • [⚠️ Юридический дисклеймер](#юридический-дисклеймер-legal-disclaimer) | • [⚠️ Legal Disclaimer](#legal-disclaimer) |
---


## Что делает программа?
Из переданного .fb2/.pdf/.txt текстового файла генерирует реалистично звучащую аудиокнигу. В том числе можно автоматически перевести и озвучить иностранную книгу на русском.

## Как это происходит?
+ **Получаем текст** и отправляем его на обработку в Gemini(ИИ от Google) написав для неё промт или используя написанные, это позволит профессионально перевести книгу с любого крупного языка, провести работу редактора(если API ключа нет, этот шаг можно пропустить)
+ **Выбираем модель для первоначальной озвучки**.

Облачная **Edge TTS**, более качественная, требует интернет. 

Локальная **Silero TTS v5**, звук заметно более роботизированный, зато умеет работать с ударениями от Gemini и после глубокой обработки на следующих шагах звучит неплохо.

+ Переозвучка на собственный голос. Это существенно повышает общее качество звучания, за счет неидеальности человеческого голоса для переозвучки, эффект робота становится меньше. Я переозвучивал на собственный голос, при установке программы вам нужно будет сгенерировать ваш голос или найти уже существующий, подробнее об этом далее. Программа позволяет это сделать.
+ Глубокая обработка голоса, с целью повысить качество звучания и убрать эффект роботизированности. Тут несколько вариантов обработки, для Edge и для Silero отдельно.

## Что получается в итоге?

Пример звучания книги в переозвучке Edge TTS/Мой голос: <a href="https://drive.google.com/file/d/1mQtj4c0SPY2yOdax7M6NsAkFxYcolBbD/view?usp=drive_link">Открыть пример озвучки</a>.

Примеры звучания Silero:

+ Silero/Aidar/Мой голос: <a href="https://drive.google.com/file/d/1uR8xUctQ76N9Ud9S3mV8Qhuc7aR1cS99/view?usp=drive_link">Открыть пример озвучки</a>. 
+ Silero/Aidar/Только чистка голоса от роботизированности <a href="https://drive.google.com/file/d/1e4HP8xYwgHehOJkn4904XaG246klK3vO/view?usp=drive_link">Открыть пример озвучки</a>. 

Далее подробное описание возможностей и описание процесса установки.

## Два режима синтеза речи, две разные нейросети.

**Edge TTS (облачный)**

Голос «Дмитрий» (ru-RU-DmitryNeural), очень реалистичное звучание на русском.

+ Параллельные запросы (до 5 одновременных) для ускорения
+ Автоматические повторные попытки при сбоях
+ Защита от блокировок через случайные задержки

**Silero TTS v5 (локальный)**
Полностью офлайн, не требует интернета

Голоса: Eugene и Aidar
Работает на GPU через PyTorch

## Интеллектуальная коррекция текста через Gemini API

В файле pipeline/parsers.py можно указать собственные промты, они будут вместе с текстом отправлены нейросетям при озвучке, в том числе можно передать список ударений и промт для него.

В базовой версии настроено:

+ Перевод на русский, если книга иностранная.
+ Исправление омографов — слов с неоднозначным ударением:

«за́мок» (дворец) / «замо́к» (устройство)

«сто́ит» (цена) / «стои́т» (находится)

+ Восстановление буквы «ё» — «еще» → «ещё»
+ Раскрытие чисел — «2025 год» → «две тысячи двадцать пятый год»
+ Расшифровка сокращений — «г. Москва» → «город Москва»

 и тд

При обработке книги в Gemini результаты кешируются, если процесс неожиданно прервется, то при повторном запуске программа начнет с последнего "сохранения".

API ключ нужно указать в файле .env , допустима ротация ключей, если их несколько.

## Преобразование голоса через RVC (Retrieval-based Voice Conversion)

RVC позволяет обучить модель на основе вашего голоса https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
Это несложно, про обучение модели в части про установку.

Или можно использовать готовую модель, пути к модели устанавливаются в pipeline/config.py

После программа автоматически будет предлагать использовать сгенерированный или просто загруженный вами голос.

Некоторые особенности:

+ Алгоритм извлечения pitch: RMVPE (высокая точность)
+ Half-precision вычисления для экономии VRAM
+ Умный контроль видеопамяти с ожиданием освобождения(программа писалась под видеокарту с 6гб памяти)

## Постобработка аудио (Humanize)

Два уровня обработки:

+ Очистка артефактов синтеза (для Silero)
+ Финальная очистка после RVC
+ Удаление цифрового нуля — замена на микрошум (-90 дБ)
+ Мягкий лимитер — предотвращение искажений
+ Нормализация — выравнивание громкости

## Надёжность и отказоустойчивость

+ Автоматическое восстановление после сбоев
+ Сохранение состояния после каждого фрагмента и блока
+ Разные стратегии повторов для разных типов ошибок
+ Глобальный флаг занятости RVC — исключение параллельных запусков
+ Подробное логирование всех этапов в process.log

## Поддержка форматов книг

+ FB2	Полное извлечение структуры через lxml, включая сноски
+ TXT	Разбиение по пустым строкам, очистка от мусора
+ PDF	Извлечение через pdfplumber, обрезка колонтитулов, выбор страниц
  
## Гибкая сборка итоговых файлов

+ Блоки по 10 минут для обработки RVC
+ Финальные MP3 до 2 часов каждый
+ Битрейт 320 kbps (максимальное качество)

Программа создает папки с файлами после:

+ Первоначальная обработка Silero/Edge +улучшение
+ Обработка RVC
+ Финальная очистка и улучшение звука.

Для более мощных систем возможны настройки в pipeline/config.py повышающие качество звука и изменяющие размеры фрагментов(использовать осторожно). 
Сейчас система оптимизирована под видеокарту с 6ГБ видеопамяти.

> [!NOTE]
> ## 🛠️ Системные требования

| Компонент | Минимальные | Рекомендуемые |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GTX 1060 (6 ГБ) | NVIDIA RTX 3060+ (12 ГБ) |
| **VRAM** | 6 ГБ | 12+ ГБ |
| **RAM** | 16 ГБ | 32 ГБ |
| **Диск** | 10 ГБ (свободно) | 20+ ГБ **SSD** |
| **Интернет** | Для *Edge TTS* и *Gemini* | Стабильное соединение |


> [!IMPORTANT]
> ## 📋 Требуемое программное обеспечение

| ПО | Версия | Назначение |
| :--- | :---: | :--- |
| **Windows** | 10 / 11 | Основная ОС |
| **Python** | `3.12.x` | **Основное окружение** |
| **Python** | `3.10.x` | Для *RVC* |
| **CUDA Toolkit** | `11.8` | Для ускорения вычислений на **GPU** |
| **FFmpeg** | `5+` | Обработка и конвертация аудио |
| **Git** | `2.x` | Клонирование репозитория и управление версиями |




> [!WARNING]
> **Внимание!** Везде в инструкции используется путь `C:\ArgonNeuroReader`. 
> Вы можете выбрать любой другой диск и папку, например `D:\Projects\ArgonNeuroReader`. Просто замените путь в командах и в файле конфигурации.

## Шаг 1: Клонирование репозитория

Откройте терминал (Командную строку или PowerShell) и выполните следующие команды:

```bash
# Переходим на диск C (или ваш целевой диск)
c:

# Клонируем репозиторий в папку ArgonNeuroReader
git clone https://github.com/Argonstudio/ArgonNeuroReader.git

# Переходим в папку проекта
cd ArgonNeuroReader
```
## Шаг 2: Создание виртуального окружения

Убедитесь, что вы находитесь внутри корневой папки проекта, и создайте изолированное окружение:

```bash
# Переходим в папку репозитория
cd C:\ArgonNeuroReader

# Создаём виртуальное окружение с именем venv
python -m venv venv

# Активируем окружение (для Windows)
venv\Scripts\activate

# Проверяем версию активного Python
python --version
```

> [!NOTE]
> В терминале слева от строки ввода должно появиться название окружения: `(venv)`.
> Успешный ответ команды проверки: `Python 3.12.x`.

---

## Шаг 3: Установка PyTorch с поддержкой CUDA

> [!IMPORTANT]
> Настоятельно рекомендуется устанавливать указанные ниже версии библиотек для полной совместимости с CUDA 11.8. Загрузка пакетов займет около 5–10 минут (общий размер порядка **2.5 ГБ**).

Убедитесь, что виртуальное окружение `(venv)` по-прежнему активно, и выполните установку:

```bash
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### Проверка корректности установки

После завершения скачивания обязательно протестируйте доступность видеокарты в PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

**Ожидаемый вывод в консоли:**
```text
True
2.5.1+cu118
```
## Шаг 4: Установка основных зависимостей

Убедитесь, что виртуальное окружение активировано `(venv)`, и запустите установку всех необходимых библиотек из файла зависимостей:

```bash
pip install -r requirements.txt
```

#### Что будет установлено в вашей системе:

*   **`edge-tts`** — Облачный синтез речи (генерация базового голоса Дмитрия).
*   **`google-generativeai`** — Интеграция с Gemini API для умной коррекции текста и перевода.
*   **`librosa`** — Анализ, обработка и подготовка аудиофайлов.
*   **`pdfplumber`** — Извлечение текста и очистка разметки из файлов формата PDF.
*   **`cupy-cuda11x`** — Аппаратное **GPU-ускорение** для алгоритмов постобработки Humanize.
*   *... а также сопутствующие утилиты для логирования, парсинга формата FB2 (`lxml`) и работы со звуком.*

> [!TIP]
> Если при установке `cupy-cuda11x` возникают ошибки компиляции, убедитесь, что в вашей системе правильно настроен и установлен **CUDA Toolkit 11.8**, как указано в системных требованиях.

## 🎙️ Шаг 5: Установка RVC (Обязательно)

Компонент **RVC (Retrieval-based Voice Conversion)** отвечает за переозвучку и замену роботизированных голосов нейросетей на ваш собственный сохраненный голос.

### 5.1. Клонирование репозитория RVC-WebUI

Убедитесь, что вы находитесь внутри корневой директории основного проекта, и склонируйте движок RVC во вложенную папку `rvc_engine`:

```bash
# Проверяем, что находимся в корне проекта
cd C:\ArgonNeuroReader

# Клонируем официальный движок RVC в подпапку rvc_engine
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git rvc_engine

# Переходим в директорию склонированного движка
cd rvc_engine
```

> [!IMPORTANT]
> Клонирование должно выполняться **строго внутри папки `C:\ArgonNeuroReader`**. 
> Команда `git clone` автоматически создаст необходимую для работы конвейера структуру папок `C:\ArgonNeuroReader\rvc_engine\`.

### 5.2. Создание виртуального окружения (Python 3.10)

> [!WARNING]
> **Критически важно!** Движок RVC требует для работы строго **Python 3.10** (основной конвейер проекта использует 3.12, но здесь версии разделены). Если у вас не установлен Python 3.10, скачайте его с официального сайта [python.org](https://python.org).

Убедитесь, что вы находитесь в папке `rvc_engine`, и изолируйте окружение:

```bash
# Создаём отдельное виртуальное окружение для RVC
py -3.10 -m venv venv310

# Активируем его
venv310\Scripts\activate

# Проверяем версию интерпретатора
python --version
```
*Успешный ответ терминала: `Python 3.10.x`.*

---

### 5.3. Установка PyTorch 2.0.1 с поддержкой CUDA

> [!IMPORTANT]
> **Не устанавливайте версии PyTorch 2.6+!** Это гарантированно вызовет критическую ошибку `UnpicklingError` на этапе инференса из-за жесткой несовместимости со старой кодовой базой библиотеки `fairseq`.

Выполните установку строго совместимой версии фреймворка:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

### 5.4. Установка зависимостей RVC

Установите стандартный набор библиотек, поставляемый с движком:

```bash
pip install -r requirements.txt
```

---

### 5.5. Установка дополнительных библиотек

Доустановите фиксы для логирования и совместимости графиков:

```bash
pip install tensorboard tensorboardX
pip install matplotlib==3.5.3
```

---

### 5.6. Деактивация окружения

После успешного развертывания ПО закройте окружение RVC:

```bash
deactivate
```

---

### 5.7. Загрузка предобученных базовых моделей RVC

Для извлечения тона и признаков голоса программе необходимы веса нейросетей-доноров. Перейдите в корневую папку движка и создайте структуру директорий:

```bash
cd C:\ArgonNeuroReader\rvc_engine

mkdir assets\hubert
mkdir assets\rmvpe
mkdir assets\pretrained_v2
```

Скачайте файлы моделей и распределите их строго по указанным путям:

| Имя файла | Размер | Куда поместить (относительно `rvc_engine/`) |
| :--- | :---: | :--- |
| **`hubert_base.pt`** | ~180 МБ | `assets/hubert/` |
| **`rmvpe.pt`** | ~325 МБ | `assets/rmvpe/` |
| **`f0G40k.pth`** | — | `assets/pretrained_v2/` |
| **`f0D40k.pth`** | — | `assets/pretrained_v2/` |

> [!TIP]
> **Где взять файлы:** Вы можете легально скачать эти веса из официального репозитория моделей на [Hugging Face lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) или найти ссылки в оригинальных релизах [RVC-Project Releases](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases).

---

### 5.8. Размещение вашей пользовательской модели (Голоса)

Если у вас есть готовый обученный голос (индекс и веса), добавьте его в систему:

```bash
# Создаём именную папку для модели голоса
mkdir logs\my_voice
```

Поместите в созданную папку `logs/my_voice/` два ваших файла:
1.  **`my_voice_e300_s2500.pth`** — файл весов модели.
2.  **`added_xxx.index`** — файл индекса признаков.

> [!NOTE]
> Если у вас пока нет готовой модели, вы можете обучить её самостоятельно через встроенный интерфейс `rvc_engine`, скачать готовые паки из сообщества.

## ⚙️ Шаг 6: Настройка конфигурации конвейера

Откройте файл `pipeline/config.py` в любом удобном текстовом редакторе (например, Блокноте, VS Code или Notepad++) и скорректируйте основные переменные под вашу систему.

### 6.1. Базовые пути
Найдите блок базовых путей и укажите актуальный путь к корневой директории вашего проекта:

```python
# ============================================================
# БАЗОВЫЕ ПУТИ — ИЗМЕНИТЕ ПОД СВОЮ СИСТЕМУ!
# ============================================================

# Корневая папка проекта
BASE_DIR = r'C:\ArgonNeuroReader'  # ← Укажите ваш путь, если он отличается

# Папка с исходной книгой
BOOK_DIR = os.path.join(BASE_DIR, 'book')

# Пути к исполняемым файлам RVC
RVC_DIR = os.path.join(BASE_DIR, 'rvc_engine')
RVC_CLI = os.path.join(RVC_DIR, 'tools', 'infer_cli.py')
RVC_PYTHON = os.path.join(RVC_DIR, 'venv310', 'Scripts', 'python.exe')
```

### 6.2. Привязка вашей RVC-модели
Прокрутите файл ниже и укажите имена файлов вашей обученной модели и индекса, которые вы разместили на Шаге 5.8:

```python
# Имя файла вашей обученной модели (веса)
MODEL_NAME = "my_voice_e300_s2500.pth"  # ← ЗАМЕНИТЕ на имя вашего файла .pth

# Абсолютный путь к индексному файлу признаков
INDEX_PATH = os.path.join(RVC_DIR, "logs", "my_voice", "added_xxx.index")  # ← ЗАМЕНИТЕ на ваш .index
```

---

## 🔑 Шаг 7: Создание файла окружения `.env`

Для работы модуля умной коррекции текста на базе нейросети Gemini вам необходимо прописать API-ключ. В корневой директории проекта `C:\ArgonNeuroReader` создайте текстовый файл с именем `.env` и добавьте в него следующую строку:

```env
# API-ключ Google Gemini (для исправления опечаток и разметки пауз)
GEMINI_API_KEY=ваш_секретный_ключ_здесь
```

> [!TIP]
> **Как бесплатно получить токен Gemini API:**
> 1. Перейдите на платформу [Google AI Studio Key Developer](https://aistudio.google.com/apikey).
> 2. Авторизуйтесь под своим Google-аккаунтом.
> 3. Нажмите кнопку **«Create API key»**.
> 4. Скопируйте сгенерированный ключ и вставьте его в файл `.env` вместо заглушки.
> 
> *Примечание: Если ключ не задан, конвейер продолжит работу в штатном режиме, но будет использовать оригинальный сырой текст книги из FB2/PDF без предварительной интеллектуальной коррекции.*

---

## 📚 Шаг 8: Подготовка папки для исходной литературы

Вернитесь в консоль (с активным главным окружением Python 3.12) и создайте специальную директорию, куда вы будете загружать книги для озвучки:

```bash
# Убедитесь, что вы находитесь в корне проекта
cd C:\ArgonNeuroReader

# Создаем целевую директорию для книг
mkdir book
```

Поместите файл вашей электронной книги (поддерживаются форматы `.fb2`, `.pdf` и просто `.txt` ) в созданную папку. Структура проекта должна выглядеть следующим образом:

```text
C:\ArgonNeuroReader\book\
└── моя_любимая_книга.fb2
```
## 🧪 Шаг 9: Проверка корректности установки

Перед первым полноценным запуском убедитесь, что все ключевые модули главного окружения Python 3.12 импортируются без ошибок связи с CUDA:

```bash
# Активируем основное окружение проекта
cd C:\ArgonNeuroReader
venv\Scripts\activate

# Запускаем экспресс-тест критических библиотек
python -c "import torch; import librosa; import edge_tts; print('OK')"
```

**Ожидаемый результат в консоли:**
```text
OK
```
*Если вместо `OK` вы видите ошибку `ModuleNotFoundError` или проблемы с DLL, вернитесь к Шагам 3 и 4 и повторите установку зависимостей.*

---

## 🚀 Шаг 10: Запуск конвейера ArgonNeuroReader

Вы можете запускать обработку книг вручную через консоль или настроить автоматический запуск.

### Вариант А. Запуск через терминал (вручную)
Убедитесь, что ваше виртуальное окружение `(venv)` активно, и выполните:

```bash
python main.py
```

### Вариант Б. Создание лаунчера для Windows (Запуск в 1 клик)
Чтобы каждый раз не открывать консоль и не вводить команды активации вручную, создайте быстрый ярлык запуска.

В корневой папке `C:\ArgonNeuroReader` создайте текстовый файл, переименуйте его в **`run.bat`** (убедитесь, что расширение файла изменилось с `.txt` на `.bat`), нажмите по нему правой кнопкой мыши -> *Изменить* и вставьте следующий код:

```bat
@echo off
chcp 65001 >nul
cd /d "%~dp0"
call ".\venv\Scripts\activate"
python main.py
pause
```

> [!TIP]
> **Что делает этот скрипт:** Команда `chcp 65001` переводит кодировку консоли Windows в UTF-8. Это необходимо, чтобы названия книг на русском языке и логи генерации текста отображались в окне корректно и без «кракозябр». Теперь для запуска проекта вам достаточно просто дважды кликнуть по файлу `run.bat`.

## 📁 Итоговая структура проекта после установки

После выполнения всех шагов инструкции корневая директория вашего проекта должна выглядеть следующим образом:

```text
C:\ArgonNeuroReader\
├── main.py                     # Главный файл запуска конвейера
├── pipeline/                   # Модули логики обработки текста и звука
│   ├── config.py               # Конфигурационный файл (настроен под ваши пути)
│   ├── common.py
│   ├── edge_tts.py
│   ├── silero_tts.py
│   └── parsers.py
├── book/                       # Директория для исходных книг (.fb2, .pdf)
│   └── моя_любимая_книга.fb2
├── output_audio/               # Результаты озвучки (создаётся автоматически)
├── rvc_engine/                 # Изолированный движок клонирования голоса RVC
│   ├── assets/                 # Базовые модели-доноры весов
│   │   ├── hubert/
│   │   │   └── hubert_base.pt
│   │   ├── rmvpe/
│   │   │   └── rmvpe.pt
│   │   └── pretrained_v2/
│   │       ├── f0G40k.pth
│   │       └── f0D40k.pth
│   ├── logs/
│   │   └── my_voice/           # Папка с вашей пользовательской моделью
│   │       ├── my_voice_e300_s2500.pth
│   │       └── added_xxx.index
│   ├── tools/
│   │   └── infer_cli.py
│   └── venv310/                # Виртуальное окружение RVC (Python 3.10)
├── venv/                       # Главное виртуальное окружение проекта (Python 3.12)
├── requirements.txt            # Список основных зависимостей проекта
├── .env                        # Файл скрытых переменных среды (API-ключи Gemini)
└── run.bat                     # Скрипт быстрого запуска для Windows в один клик
```

---

## 👤 Автор

**Ivan Voitkov**

- 🌐 Сайт: [argon-studio.ru](https://argon-studio.ru/)
- 💻 GitHub: [ArgonStudio](https://github.com/Argonstudio)
- 📦 Репозиторий: [ArgonNeuroReader](https://github.com/Argonstudio/ArgonNeuroReader)

---

## 🙏 Благодарности

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — преобразование голоса
- [edge-tts](https://github.com/rany2/edge-tts) — синтез речи
- [Silero TTS](https://github.com/snakers4/silero-models) — локальный синтез
- [Google Gemini](https://ai.google.dev/) — коррекция текста
- [librosa](https://librosa.org/) — обработка аудио
- [pdfplumber](https://github.com/jsvine/pdfplumber) — парсинг PDF

---

## ⚠️ Юридический дисклеймер (Legal Disclaimer)

### Авторские права на контент
**ArgonNeuroReader** предназначен исключительно для озвучивания книг и текстов, **на которые у вас есть законные права**. Это могут быть:
- Книги в общественном достоянии (public domain)
- Книги, на которые вы приобрели права на озвучивание
- Собственные произведения
- Книги с разрешения правообладателя

Озвучивание и распространение аудиокниг, защищённых авторским правом, без разрешения правообладателя является **нарушением законодательства** и может повлечь юридическую ответственность.

### Использование сторонних сервисов
Программа использует следующие сторонние сервисы и компоненты:

#### Microsoft Edge TTS (неофициальный API)
- Библиотека `edge-tts` обращается к **неофициальным** endpoint'ам Microsoft (`speech.platform.bing.com`).
- Эти endpoint'ы **не предназначены** для публичного использования и не входят в официальный Azure Speech Services API.
- Использование может **нарушать Условия обслуживания Microsoft** (Microsoft Terms of Service).
- Microsoft может в любой момент **заблокировать доступ**, изменить API или ввести плату.
- **Автор программы не несёт ответственности** за блокировки, изменения API или юридические последствия использования неофициального API.

#### Google Gemini API
- Используется официальный API Google Gemini через `google-generativeai` SDK.
- Подчиняется [Условиям использования Google AI Studio](https://ai.google.dev/terms) и [Политике допустимого использования](https://ai.google.dev/terms/aup).
- Бесплатный тариф имеет ограничения по количеству запросов.

#### RVC (Retrieval-based Voice Conversion)
- RVC-Project распространяется под **MIT License** — свободное использование разрешено.
- Однако **клонирование голосов реальных людей** без их согласия может нарушать:
  - Права на изображение и голос
  - Законы о персональных данных (GDPR, 152-ФЗ)
  - Авторские права на аудиозаписи, использованные для обучения
- **Ответственность** за использование клонированных голосов лежит **на пользователе**.

### Отказ от гарантий
Программа предоставляется **«как есть» (AS IS)** без каких-либо гарантий, явных или подразумеваемых, включая, но не ограничиваясь:
- Гарантии товарной пригодности
- Гарантии пригодности для конкретной цели
- Гарантии ненарушения прав третьих лиц

### Ограничение ответственности
Автор программы **не несёт ответственности** за:
- Любые прямые или косвенные убытки
- Потерю данных
- Блокировку аккаунтов Microsoft/Google
- Юридические последствия использования программы
- Нарушение авторских прав пользователем

### Согласие с условиями
Используя **ArgonNeuroReader**, вы подтверждаете, что:
1. Понимаете и принимаете все вышеуказанные риски
2. Имеете законные права на озвучиваемые тексты
3. Используете голосовые модели с согласия владельцев голосов
4. Соблюдаете условия использования всех сторонних сервисов

### Рекомендации по законному использованию
- ✅ Озвучивайте книги в общественном достоянии
- ✅ Используйте собственные тексты
- ✅ Получайте разрешение правообладателей
- ✅ Используйте собственный голос для RVC
- ✅ Для коммерческого использования Edge TTS рассмотрите официальный [Azure Speech Services](https://azure.microsoft.com/services/cognitive-services/text-to-speech/)
- ❌ Не распространяйте озвученные книги без прав
- ❌ Не клонируйте голоса без согласия
- ❌ Не обходите технические ограничения сервисов

---

**Используя программу, вы принимаете все риски и соглашаетесь с условиями.**

# ENGLISH VERSION

**ArgonNeuroReader** is a console-based Python application for text-to-speech synthesis and audiobook generation.

## What does the program do?
It generates a realistic-sounding audiobook from an input `.fb2` / `.pdf` / `.txt` text file. It can also automatically translate and narrate a foreign-language book in Russian.

## How does it work?
+ **Text extraction and processing** — The text is sent to Gemini (Google AI) for editing using either custom or built-in prompts. This allows professional translation from any major language and editorial refinement. *(If no API key is available, this step can be skipped.)*
+ **Initial voice synthesis model selection**:

    - **Cloud-based Edge TTS**: Higher quality, requires internet connection.

    - **Local Silero TTS v5**: More robotic sound, but supports stress marks generated by Gemini and sounds quite good after the deep post-processing applied in the next stages.
+ **Voice cloning with your own voice** — This significantly enhances overall audio quality; the imperfections of a human voice used for cloning make the robotic effect much less noticeable. I cloned my own voice. During installation, you will need to generate your own voice model or find an existing one. The program supports this process.
+ **Deep audio post-processing** — Aimed at improving sound quality and removing the robotic effect. Several processing options are available, separately for Edge and Silero.

## What is the final result?

Example of a book narrated using Edge TTS / My voice clone: <a href="https://drive.google.com/file/d/1mQtj4c0SPY2yOdax7M6NsAkFxYcolBbD/view?usp=drive_link">Open audio sample</a>.

Silero audio samples:

+ Silero / Aidar / My voice clone: <a href="https://drive.google.com/file/d/1uR8xUctQ76N9Ud9S3mV8Qhuc7aR1cS99/view?usp=drive_link">Open audio sample</a>.
+ Silero / Aidar / Voice cleaning only (robotic effect removal): <a href="https://drive.google.com/file/d/1e4HP8xYwgHehOJkn4904XaG246klK3vO/view?usp=drive_link">Open audio sample</a>.

Below is a detailed description of features and the installation process.

## Two speech synthesis modes, two different neural networks

**Edge TTS (cloud-based)**

Voice: "Dmitry" (ru-RU-DmitryNeural), highly realistic Russian speech.

+ Parallel requests (up to 5 simultaneously) for faster processing
+ Automatic retry on failure
+ Blocking protection through randomized delays

**Silero TTS v5 (local)**
Fully offline, no internet required.

Voices: Eugene and Aidar.
Runs on GPU via PyTorch.

## Intelligent text correction via Gemini API

Custom prompts can be specified in `pipeline/parsers.py`; they are sent to the neural network along with the text during narration. You can also provide a custom stress-mark list and a prompt for it.

The default setup includes:

+ Translation into Russian for foreign-language books.
+ Disambiguation of homographs — words with ambiguous stress.

+ Restoration of the letter «ё» — «еще» → «ещё»
+ Expansion of numerals — «2025 год» → «две тысячи двадцать пятый год»
+ Expansion of abbreviations — «г. Москва» → «город Москва»

…and more.

Gemini processing results are cached. If the process is unexpectedly interrupted, the next launch resumes from the last saved checkpoint.

The API key must be specified in the `.env` file; key rotation is supported when multiple keys are provided.

## Voice conversion via RVC (Retrieval-based Voice Conversion)

RVC allows you to train a model based on your own voice: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
It is not difficult; model training is described in the installation section.

Alternatively, you can use a pre-trained model. Model paths are set in `pipeline/config.py`.

Afterwards, the program will automatically suggest using your generated or downloaded voice.

Some features:

+ Pitch extraction algorithm: RMVPE (high accuracy)
+ Half-precision computation to save VRAM
+ Smart VRAM monitoring with wait for memory release (the program was developed for a GPU with 6 GB of VRAM)

## Audio post-processing (Humanize)

Two processing levels:

+ Removal of synthesis artifacts (for Silero)
+ Final cleanup after RVC
+ Removal of digital zero — replaced with micro-noise (-90 dB)
+ Soft limiter — prevents distortion
+ Normalization — volume leveling

## Reliability and fault tolerance

+ Automatic recovery after failures
+ State saving after each fragment and block
+ Different retry strategies for different error types
+ Global RVC busy flag — prevents parallel launches
+ Detailed logging of all stages into `process.log`

## Supported book formats

+ FB2 — Full structure extraction via lxml, including footnotes
+ TXT — Splitting by blank lines, cleaning of artifacts
+ PDF — Extraction via pdfplumber, header/footer cropping, page selection

## Flexible final file assembly

+ 10-minute blocks for RVC processing
+ Final MP3 files up to 2 hours each
+ 320 kbps bitrate (maximum quality)

The program creates folders with files after each stage:

+ Initial Silero/Edge synthesis + enhancement
+ RVC processing
+ Final cleanup and audio enhancement

For more powerful systems, `pipeline/config.py` offers settings that increase audio quality and change fragment sizes (use with caution).
Currently, the system is optimized for a GPU with 6 GB of VRAM.

> [!NOTE]
> ## 🛠️ System requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GTX 1060 (6 GB) | NVIDIA RTX 3060+ (12 GB) |
| **VRAM** | 6 GB | 12+ GB |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 10 GB free | 20+ GB **SSD** |
| **Internet** | Required for *Edge TTS* and *Gemini* | Stable connection |


> [!IMPORTANT]
> ## 📋 Required software

| Software | Version | Purpose |
| :--- | :---: | :--- |
| **Windows** | 10 / 11 | Primary OS |
| **Python** | `3.12.x` | **Main environment** |
| **Python** | `3.10.x` | For *RVC* |
| **CUDA Toolkit** | `11.8` | GPU acceleration |
| **FFmpeg** | `5+` | Audio processing and conversion |
| **Git** | `2.x` | Repository cloning and version control |


> [!WARNING]
> **Attention!** This guide uses the path `C:\ArgonNeuroReader`.
> You can choose any other drive and folder, for example `D:\Projects\ArgonNeuroReader`. Just replace the path in the commands and in the configuration file.

## 🚀 Step 1: Cloning the Repository

Open your terminal (Command Prompt or PowerShell) and run the following commands:

```bash
# Switch to drive C (or your target drive)
c:

# Clone the repository into the ArgonNeuroReader folder
git clone https://github.com/Argonstudio/ArgonNeuroReader.git

# Navigate into the project folder
cd ArgonNeuroReader
```

---

## 🐍 Step 2: Creating a Virtual Environment

Make sure you are inside the root folder of the project, then create an isolated environment:

```bash
# Navigate to the repository folder
cd C:\ArgonNeuroReader

# Create a virtual environment named venv
python -m venv venv

# Activate the environment (for Windows)
venv\Scripts\activate

# Check the active Python version
python --version
```

> [!NOTE]
> Once activated, the environment name `(venv)` should appear on the left side of your terminal prompt.
> The expected successful output of the version check command is: `Python 3.12.x`.

---

## 📦 Step 3: Installing PyTorch with CUDA Support

> [!IMPORTANT]
> It is highly recommended to install the exact library versions specified below for full compatibility with CUDA 11.8. The download will take about 5–10 minutes (total size is around **2.5 GB**).

Make sure your virtual environment `(venv)` is still active, and run the installation:

```bash
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://pytorch.org
```

### Verification

Once the download is complete, make sure to test whether your GPU is accessible within PyTorch:

```bash
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Expected terminal output:**
```text
True
2.5.1+cu118
```

---

## 🛠️ Step 4: Installing Core Dependencies

Make sure the virtual environment `(venv)` is active, and trigger the installation of all required packages from the dependency file:

```bash
pip install -r requirements.txt
```

### What will be installed in your system:

*   **`edge-tts`** — Cloud-based text-to-speech synthesis (generates the base Dmitry voice).
*   **`google-generativeai`** — Integration with Gemini API for smart text correction and processing.
*   **`librosa`** — Audio file analysis, processing, and preparation.
*   **`pdfplumber`** — Text extraction and layout cleaning from PDF files.
*   **`cupy-cuda11x`** — Hardware **GPU acceleration** for the Humanize post-processing algorithms.
*   *... as well as utility packages for logging, FB2 parsing (`lxml`), and sound management.*

> [!TIP]
> If you encounter compilation errors while installing `cupy-cuda11x`, ensure that **CUDA Toolkit 11.8** is correctly installed and configured in your system environment variables, as specified in the System Requirements.

---

## 🎙️ Step 5: Installing RVC (Required)

The **RVC (Retrieval-based Voice Conversion)** component is responsible for voice cloning and replacing robotic AI voices with your own saved custom voice.

### 5.1. Cloning the RVC-WebUI Repository

Make sure you are inside the root directory of the main project, and clone the RVC engine into the nested `rvc_engine` folder:

```bash
# Verify you are in the project root
cd C:\ArgonNeuroReader

# Clone the official RVC engine into the rvc_engine subfolder
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git rvc_engine

# Navigate into the cloned engine directory
cd rvc_engine
```

> [!IMPORTANT]
> The cloning process must be performed **strictly inside the `C:\ArgonNeuroReader` folder**. 
> The `git clone` command will automatically generate the required directory structure: `C:\ArgonNeuroReader\rvc_engine\`.

### 5.2. Creating a Virtual Environment (Python 3.10)

> [!WARNING]
> **Crucial!** The RVC engine strictly requires **Python 3.10** to run (the main project pipeline uses 3.12, so the environments are kept separate). If you don't have Python 3.10 installed, download it from the official website [python.org](https://python.org).

Make sure you are in the `rvc_engine` folder, and isolate the environment:

```bash
# Create a separate virtual environment for RVC
py -3.10 -m venv venv310

# Activate it
venv310\Scripts\activate

# Verify the interpreter version
python --version
```
*Expected terminal output: `Python 3.10.x`.*

### 5.3. Installing PyTorch 2.0.1 with CUDA Support

> [!IMPORTANT]
> **Do not install PyTorch versions 2.6+!** This is guaranteed to trigger a critical `UnpicklingError` during the inference phase due to hard incompatibility with the legacy codebase of the `fairseq` library.

Install the strictly compatible version of the framework:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 5.4. Installing RVC Dependencies

Install the standard set of libraries bundled with the engine:

```bash
pip install -r requirements.txt
```

### 5.5. Installing Additional Libraries

Install fixes for logging and chart compatibility:

```bash
pip install tensorboard tensorboardX
pip install matplotlib==3.5.3
```

### 5.6. Deactivating the Environment

Once the software deployment is successful, close the RVC environment:

```bash
deactivate
```

### 5.7. Downloading Pre-trained Base RVC Models

To extract pitch and voice features, the program requires the weights of donor neural networks. Navigate to the root folder of the engine and create the following directory structure:

```bash
cd C:\ArgonNeuroReader\rvc_engine

mkdir assets\hubert
mkdir assets\rmvpe
mkdir assets\pretrained_v2
```

Download the model files and place them strictly according to the paths below:

| File Name | Size | Where to place (relative to `rvc_engine/`) |
| :--- | :---: | :--- |
| **`hubert_base.pt`** | ~180 MB | `assets/hubert/` |
| **`rmvpe.pt`** | ~325 MB | `assets/rmvpe/` |
| **`f0G40k.pth`** | — | `assets/pretrained_v2/` |
| **`f0D40k.pth`** | — | `assets/pretrained_v2/` |

> [!TIP]
> **Where to get the files:** You can legally download these weights from the official model repository on [Hugging Face lj1995/VoiceConversionWebUI](https://huggingface.co) or find the links in the original [RVC-Project Releases](https://github.com).

### 5.8. Placing Your Custom Trained Model (Voice)

If you have a ready-made trained voice (index and weights), add it to the system:

```bash
# Create a named folder for the voice model
mkdir logs\my_voice
```

Place your two files into the newly created `logs/my_voice/` folder:
1.  **`my_voice_e300_s2500.pth`** — the model weights file.
2.  **`added_xxx.index`** — the feature index file.

> [!NOTE]
> If you don't have a ready-made model yet, you can train one yourself via the built-in `rvc_engine` UI, download pre-made packs from the community, or run the `ArgonNeuroReader` pipeline without the RVC step (in this case, the voice will remain the base one from Edge TTS/Silero, but the RVC engine must still be correctly installed in the system).

---

## ⚙️ Step 6: Configuring the Pipeline

Open the `pipeline/config.py` file in any text editor (such as Notepad, VS Code, or Notepad++) and adjust the main variables to match your system.

### 6.1. Base Paths
Locate the base paths block and specify the correct path to your project's root directory:

```python
# ============================================================
# BASE PATHS — MODIFY ACCORDING TO YOUR SYSTEM!
# ============================================================

# Project root folder
BASE_DIR = r'C:\ArgonNeuroReader'  # ← REPLACE with your path if different

# Folder for source books (auto-generated)
BOOK_DIR = os.path.join(BASE_DIR, 'book')

# Paths to RVC executables
RVC_DIR = os.path.join(BASE_DIR, 'rvc_engine')
RVC_CLI = os.path.join(RVC_DIR, 'tools', 'infer_cli.py')
RVC_PYTHON = os.path.join(RVC_DIR, 'venv310', 'Scripts', 'python.exe')
```

### 6.2. Linking Your RVC Model
Scroll down the file and specify the file names of your trained model and index that you placed in Step 5.8:

```python
# File name of your trained model (weights)
MODEL_NAME = "my_voice_e300_s2500.pth"  # ← REPLACE with your .pth file name

# Absolute path to the feature index file
INDEX_PATH = os.path.join(RVC_DIR, "logs", "my_voice", "added_xxx.index")  # ← REPLACE with your .index file path
```

---
---

## 🔑 Step 7: Creating the `.env` Environment File

To enable the smart text correction module powered by the Gemini neural network, you need to configure an API key. In the root directory of the project (`C:\ArgonNeuroReader`), create a text file named `.env` and add the following line:

```env
# Google Gemini API key (for typo corrections and pause layout formatting)
GEMINI_API_KEY=your_secret_key_here
```

> [!TIP]
> **How to get a Gemini API token for free:**
> 1. Go to the [Google AI Studio Key Developer](https://google.com) platform.
> 2. Log in using your Google account.
> 3. Click the **"Create API key"** button.
> 4. Copy the generated key and paste it into the `.env` file, replacing the placeholder.
> 
> *Note: If the key is not set, the pipeline will continue to work normally, but it will process the raw text of the book from FB2/PDF without prior intelligent text correction.*

---

## 📚 Step 8: Preparing the Books Folder

Go back to the console (with the main Python 3.12 environment active) and create a dedicated directory where you will upload books for text-to-speech processing:

```bash
# Ensure you are in the project root
cd C:\ArgonNeuroReader

# Create the target directory for books
mkdir book
```

Place your e-book files (supported formats are `.fb2` and `.pdf`) into the created folder. The project structure should look like this:

```text
C:\ArgonNeuroReader\book\
└── my_favorite_book.fb2
```

---

## 🧪 Step 9: Verifying the Installation

Before running the pipeline for the first time, verify that all core modules of the main Python 3.12 environment import correctly without CUDA-related errors:

```bash
# Activate the main project environment
cd C:\ArgonNeuroReader
venv\Scripts\activate

# Run a quick check of the critical libraries
python -c "import torch; import librosa; import edge_tts; print('OK')"
```

**Expected terminal output:**
```text
OK
```
*If you see a `ModuleNotFoundError` or DLL issues instead of `OK`, return to Steps 3 and 4 and reinstall the dependencies.*

---

## 🚀 Step 10: Running the ArgonNeuroReader Pipeline

You can run the book processing manually via the console or configure an automated launcher shortcut.

### Option A. Launch via Terminal (Manual)
Make sure your virtual environment `(venv)` is active and run:

```bash
python main.py
```

### Option B. Creating a Windows Launcher (1-Click Run)
To avoid opening the console and typing activation commands manually every time, you can create a quick launch shortcut.

In the root folder `C:\ArgonNeuroReader`, create a new text file and rename it to **`run.bat`** (make sure the file extension changes from `.txt` to `.bat`). Right-click it -> *Edit*, and paste the following code:

```bat
@echo off
chcp 65001 >nul
cd /d "%~dp0"
call ".\venv\Scripts\activate"
python main.py
pause
```

> [!TIP]
> **What this script does:** The `chcp 65001` command switches the Windows console encoding to UTF-8. This is necessary so that Cyrillic book titles and generation logs are displayed correctly in the window without encoding glitches or corrupted symbols. Now you can launch the project simply by double-clicking the `run.bat` file.

---

## 📁 Final Project Structure After Installation

Once all steps of the guide are completed, your root project directory should match the layout below:

```text
C:\ArgonNeuroReader\
├── main.py                     # Main script to run the pipeline
├── pipeline/                   # Modules handling text and audio logic
│   ├── config.py               # Configuration file (tailored to your paths)
│   ├── common.py
│   ├── edge_tts.py
│   ├── silero_tts.py
│   └── parsers.py
├── book/                       # Directory for source e-books (.fb2, .pdf)
│   └── my_favorite_book.fb2
├── output_audio/               # Generated audiobooks output (auto-created)
├── rvc_engine/                 # Isolated RVC voice cloning engine
│   ├── assets/                 # Base pre-trained model weights
│   │   ├── hubert/
│   │   │   └── hubert_base.pt
│   │   ├── rmvpe/
│   │   │   └── rmvpe.pt
│   │   └── pretrained_v2/
│   │       ├── f0G40k.pth
│   │       └── f0D40k.pth
│   ├── logs/
│   │   └── my_voice/           # Folder containing your custom voice model
│   │       ├── my_voice_e300_s2500.pth
│   │       └── added_xxx.index
│   ├── tools/
│   │   └── infer_cli.py
│   └── venv310/                # RVC virtual environment (Python 3.10)
├── venv/                       # Main project virtual environment (Python 3.12)
├── requirements.txt            # List of core project dependencies
├── .env                        # Environment file for secret tokens (Gemini API key)
└── run.bat                     # Windows batch script for quick 1-click execution
```

---

## 👤 Author

**Ivan Voitkov**

- 🌐 Website: [argon-studio.ru](https://argon-studio.ru/)
- 💻 GitHub: [ArgonStudio](https://github.com/Argonstudio)
- 📦 Repository: [ArgonNeuroReader](https://github.com/Argonstudio/ArgonNeuroReader)

---

## 🙏 Acknowledgements

We express our gratitude to the creators and maintainers of the core technologies that made this project possible:

*   **[RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** — AI voice conversion and cloning engine.
*   **[edge-tts](https://github.com/rany2/edge-tts)** — Cloud-based high-quality speech synthesis wrapper.
*   **[Silero TTS](https://github.com/snakers4/silero-models)** — High-performance local text-to-speech models.
*   **[Google Gemini](https://ai.google.dev/)** — Advanced LLM API used for text structure and typo correction.
*   **[librosa](https://librosa.org/)** — Python library for audio and music analysis.
*   **[pdfplumber](https://github.com/jsvine/pdfplumber)** — Precise text extraction from PDF documents.

---

## ⚠️ Legal Disclaimer

### Content Copyright
**ArgonNeuroReader** is intended solely for narrating books and texts **to which you own the legal rights**. This includes:
- Books in the public domain
- Books for which you have purchased audiobook narration rights
- Your own original works
- Books used with the copyright holder's permission

Narrating and distributing audiobooks protected by copyright without the copyright holder's permission is a **violation of the law** and may result in legal liability.

### Use of Third-Party Services
The program uses the following third-party services and components:

#### Microsoft Edge TTS (Unofficial API)
- The `edge-tts` library accesses **unofficial** Microsoft endpoints (`speech.platform.bing.com`).
- These endpoints are **not intended** for public use and are not part of the official Azure Speech Services API.
- Usage may **violate Microsoft's Terms of Service**.
- Microsoft may **block access**, change the API, or introduce fees at any time.
- **The author of the program is not responsible** for blocks, API changes, or legal consequences arising from the use of the unofficial API.

#### Google Gemini API
- The official Google Gemini API is used via the `google-generativeai` SDK.
- It is subject to the [Google AI Studio Terms of Service](https://ai.google.dev/terms) and the [Acceptable Use Policy](https://ai.google.dev/terms/aup).
- The free tier has request limitations.

#### RVC (Retrieval-based Voice Conversion)
- RVC-Project is distributed under the **MIT License** — free use is permitted.
- However, **cloning the voices of real people** without their consent may violate:
  - Personality and voice rights
  - Personal data protection laws (GDPR, etc.)
  - Copyrights of audio recordings used for training
- **Responsibility** for the use of cloned voices lies **with the user**.

### Disclaimer of Warranties
The program is provided **"AS IS"** without any warranties, express or implied, including but not limited to:
- Warranties of merchantability
- Warranties of fitness for a particular purpose
- Warranties of non-infringement of third-party rights

### Limitation of Liability
The author of the program **is not liable** for:
- Any direct or indirect damages
- Data loss
- Microsoft/Google account suspension
- Legal consequences of using the program
- Copyright infringement by the user

### Acceptance of Terms
By using **ArgonNeuroReader**, you confirm that you:
1. Understand and accept all the risks described above
2. Hold the legal rights to the texts being narrated
3. Use voice models with the consent of the voice owners
4. Comply with the terms of service of all third-party services

### Recommendations for Lawful Use
- ✅ Narrate books in the public domain
- ✅ Use your own original texts
- ✅ Obtain permission from copyright holders
- ✅ Use your own voice for RVC cloning
- ✅ For commercial use of Edge TTS, consider the official [Azure Speech Services](https://azure.microsoft.com/services/cognitive-services/text-to-speech/)
- ❌ Do not distribute narrated books without proper rights
- ❌ Do not clone voices without consent
- ❌ Do not circumvent technical limitations of third-party services

---

**By using this program, you accept all risks and agree to the terms.**

---

**Ready to roll!** If you encounter any bugs, crashes, or installation issues, please feel free to open an **Issue** in the repository.
