@echo off
:: Переключаем кодировку консоли на UTF-8
chcp 65001 >nul
cd /d "C:\newApp\books\audio_project"

echo [1/2] Активация виртуального окружения...
call ".\venv\Scripts\activate"

:: Настройка прокси для Edge-TTS и Gemini
set HTTP_PROXY=http://127.0.0.1:10808
set HTTPS_PROXY=http://127.0.0.1:10808
set ALL_PROXY=http://127.0.0.1:10808
echo [i] Трафик направлен через прокси 127.0.0.1:10808

echo [2/2] Запуск скрипта...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Произошла критическая ошибка выполнения.
)
echo.
echo Нажмите любую клавишу, чтобы закрыть это окно.
pause
