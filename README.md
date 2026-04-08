# meet2task

Транскрипция аудио и видео в структурированный текст: **[Groq](https://groq.com) Whisper**, постобработка через **Ollama**, опционально — **Telegram-бот** для приёма файлов и ссылок (Яндекс.Диск / Google Drive для файлов больше лимита бота).

## Возможности

- Транскрипция локальных файлов и по URL (Groq Whisper Large v3 Turbo).
- Постобработка транскрипта локальной LLM по промпту из каталога `prompt/`.
- Telegram-бот: голосовые, видео-заметки, документы; команды для реестра диалогов (SQLite в `data/`).
- Наблюдатель каталога `drop/` для пакетной обработки.

## Требования

- Python **3.10+**
- Ключ [Groq API](https://console.groq.com/keys), токен [Telegram Bot](https://t.me/BotFather) (для бота)
- [Ollama](https://ollama.com) с выбранной моделью (для постобработки)

## Установка

Клонируйте репозиторий и установите пакет в режиме разработки (код в `src/meet2task/`):

```bash
pip install -e .
```

Либо только зависимости:

```bash
pip install -r requirements.txt
```

Секреты и пути задаются в **`.env`** и **`config.yaml`** в **корне клона** (не внутри `src/`).

## Конфигурация

1. Скопируйте `config.example.yaml` → `config.yaml`.
2. Скопируйте `.env.example` → `.env` и укажите `GROQ_API_KEY` и `TELEGRAM_BOT_TOKEN` (для бота).

Переопределение каталога проекта (если запускаете скрипты не из корня репозитория):

```bash
set MEET2TASK_ROOT=D:\path\to\meet2task
```

## Запуск

После `pip install -e .` доступны команды:

| Команда | Описание |
|--------|----------|
| `meet2task-bot` | Telegram-бот |
| `meet2task-watch` | Наблюдение за `drop/` |
| `meet2task-transcribe` | CLI транскрипции |

Без установки пакета можно вызывать скрипты из корня репозитория (они подключают `src/`):

```bash
python telegram_voice_bot.py
python watch_drop.py
python webm_to_txt.py audio.webm -o out.txt
```

### Telegram

Файлы **больше 20 МБ** (лимит Bot API): включите в `config.yaml` опции `telegram.yandex_disk_from_url` и `telegram.google_drive_from_url`, выложите файл на **Яндекс.Диск** или **Google Drive** с публичным доступом и отправьте боту **сообщение со ссылкой**.

### Drop-папка

```bash
meet2task-watch
meet2task-watch -d /путь/к/папке
meet2task-watch --no-move
```

## Структура репозитория

```
src/meet2task/     # пакет Python
  config.py        # загрузка config.yaml / .env
  transcription.py # Groq Whisper
  telegram_bot.py  # бот
  ...
prompt/            # промпты для Ollama
config.example.yaml
.env.example
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
