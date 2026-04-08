# PipelineOpt

Модуль трансформации аудио (.webm и др.) в структурированный txt через Groq Whisper Large v3 Turbo и локальную LLM, работающий через Telegram-бот.

## Компоненты

- **webm_to_txt.py** — транскрипция аудио через Groq Whisper Large v3 Turbo
- **telegram_voice_bot.py** — Telegram-бот для приёма голосовых/видео/документов
- **localLLMmanager.py** — обработка транскрипции локальной LLM (Ollama)
- **prompt/process_transcription.txt** — промпт для обработки транскрипции

## Установка

```bash
pip install -r requirements.txt
```

## Конфигурация

1. Скопируйте `config.example.yaml` в `config.yaml` (пути, Ollama и т.д.).
2. Секреты задайте в файле **`.env`** в корне проекта: скопируйте [`.env.example`](.env.example) в `.env` и укажите:
   - **`GROQ_API_KEY`** — [Groq API Key](https://console.groq.com/keys)
   - **`TELEGRAM_BOT_TOKEN`** — токен от [@BotFather](https://t.me/BotFather)

   Альтернатива: те же значения можно прописать в `config.yaml` в `groq.api_key` и `telegram.bot_token` (файл не должен попадать в git).

Переменные окружения `GROQ_API_KEY` и `TELEGRAM_BOT_TOKEN` переопределяют значения из `.env` и `config.yaml`.

## Запуск

### Telegram-бот

```bash
python telegram_voice_bot.py
```

Отправьте боту голосовое сообщение, видео-заметку или документ (.webm, .ogg, .mp3, .wav).

**Файлы больше 20 МБ (лимит Telegram):** в `config.yaml` включены `telegram.yandex_disk_from_url` и `telegram.google_drive_from_url`. Выложите файл на **Яндекс.Диск** или **Google Drive** с публичным доступом по ссылке и отправьте боту **одно сообщение со ссылкой**. Для **Google Drive** подойдёт ссылка на **файл** или на **папку** (берётся первый подходящий аудио/видео файл). Groq получит прямую ссылку и обойдёт лимит бота.

Бот:
1. Транскрибирует аудио через Groq Whisper
2. Обработает текст локальной LLM (Ollama)
3. Сохранит результат в `output/` и отправит вам

### Требования

- Ollama с моделью (например, `gpt-oss:20b`) запущен локально
- Groq API ключ
- Telegram Bot Token

### Автообработка из drop-папки

Положите видео/аудио в папку `drop/` (по умолчанию) и запустите:

```bash
python watch_drop.py
```

Скрипт будет опрашивать папку и автоматически обрабатывать новые файлы. Обработанные файлы перемещаются в `drop/processed/`.

```bash
python watch_drop.py -d /путь/к/папке   # другая директория
python watch_drop.py --no-move          # не перемещать после обработки
```

### Только транскрипция (CLI)

```bash
python webm_to_txt.py audio.webm -o output.txt
python webm_to_txt.py audio.webm --no-save  # только вывод в консоль
```
