# 🇺🇦 Zoom Live UA Subtitles

Free, real-time **English → Ukrainian subtitles** for a Zoom webinar.

## What is it for?
You are running a Zoom webinar. The lecturer speaks English, but many listeners don't understand English.
This app runs on the host's Windows PC, **listens to the Zoom sound**, turns the English speech into text,
**translates it into Ukrainian**, and shows the Ukrainian subtitles **live to every participant**.

- **100% free**: no paid Zoom plan, no paid services. Speech recognition runs locally on your PC.
- **No words lost**: every sentence is transcribed and translated, in order. A few seconds of delay is normal.
- **Nothing for participants to install**: they just open a link in any browser (PC or phone).
- **Two ways to see subtitles:**
  1. **A web page** (link you post in the Zoom chat): big subtitles, full history, search, downloads, custom fonts/colors, and a floating "Поверх Zoom" window that sits on top of Zoom like movie subtitles.
  2. **Zoom's own caption bar** (optional): Ukrainian captions appear inside Zoom when participants turn on CC.
- **Everything is saved**: each session is stored as Ukrainian text, English text, bilingual text and an `.srt` subtitle file.

## How it works
Zoom sound → speech detection → English text (Whisper AI on your PC) → Ukrainian translation (Gemini AI, or a free backup) → participants' web page + Zoom captions + saved files.

---

## Requirements
- Windows 10 or 11
- Internet connection
- About 3 GB of free disk space (first run only)
- NVIDIA graphics card recommended (faster). It also works without one, just slower.

---

## Quick start (host)

### First time only
1. Unzip the folder somewhere simple, e.g. `C:\ZoomSubs` (no Cyrillic letters or spaces in the path).
2. Double-click **`START.bat`**.
   - If Windows says "Windows protected your PC" → click **More info → Run anyway**.
   - The first run installs everything automatically (5–15 min). Just wait.
3. The **control panel** opens in your browser. A setup wizard appears:
   - **Gemini key (recommended, free):** go to https://aistudio.google.com/apikey → sign in with Google → **Create API key** → copy → paste. You can skip it; a free backup translator is used instead (lower quality).
   - **Sound device:** keep the first option ("Системний звук — типовий пристрій виводу").
   - **Test:** click the test button. Ukrainian lines should appear.
   - Click **Завершити ✓**.

### Before every lecture (5 minutes early)
1. Double-click **`START.bat`** (fast from now on). Keep the black window open the whole time.
2. Wait until all status dots are **green** (Speech recognition, Translator, Public link).
3. Press the big green **▶ СТАРТ** button.
4. Click **💬 "Копіювати повідомлення для чату"** and paste it into the Zoom chat.
   Participants get the link plus instructions.
5. Join/keep Zoom open **on the same PC** with sound playing (headphones are fine).
   The app captures whatever the PC plays.

### During the lecture
- Watch the **sound level bar**: it should move when the lecturer speaks.
- **"Останні рядки"** shows the English and the Ukrainian live.
- "Delay" = seconds from speech to subtitle (a few seconds is normal).
- "Queue" may grow briefly. Nothing is lost; it catches up.

### After the lecture
- Press **■ СТОП**, then close the black window.
- Find all subtitles in **`transcripts/<date_time>/`**, or click **📂 Відкрити папку із записами** in the panel.

---

## For participants (paste into the Zoom chat)
> 🇺🇦 Українські субтитри: відкрийте посилання → субтитри з'являться автоматично.
> Натисніть на субтитри — побачите всю історію · ⚙ — налаштування (шрифт, кольори)
> «Поверх Zoom» — субтитри плавають над вікном Zoom (Chrome/Edge на комп'ютері)
> На телефоні — кнопка «На весь екран».

What participants can do on the page:
- **Read live subtitles** (large, last 1–3 lines).
- **Click the subtitles** to see the full history, search it, switch between Ukrainian only or EN + UA, copy everything, or download `.txt` / `.srt`.
- **⚙ Settings:** font size, boldness, colors, background opacity, number of lines, position, show English, dark/light theme, and quick presets (Кіно, Великий шрифт, Контрастний, Мінімал).
- **📺 Поверх Zoom:** a small floating subtitle window that stays on top of Zoom (best in Chrome/Edge on a computer).
- **⛶ На весь екран:** big subtitles; great for a phone or tablet placed next to the laptop.

---

## Optional: Ukrainian captions inside Zoom itself
1. On zoom.us → **Settings → Meeting → In Meeting (Advanced)** → turn on **Closed captioning** and **"Allow use of caption API Token to integrate with 3rd-party Closed Captioning services"**.
2. In the live meeting/webinar, the host clicks **Captions / CC → Set up manual captioner** (or "Third-party captioning service") → **Copy the API token**.
3. Paste it into the control panel field **"URL / API-токен субтитрів Zoom"** → click **Перевірити** → ✅.
4. Participants turn on **CC / Subtitles** in Zoom.

Menu names can differ slightly between Zoom versions. If this option isn't available on your account, the web link still works.

---

## Glossary (names and special terms)
In the control panel → **📖 Глосарій**: English term on the left, Ukrainian on the right.
Leave the right side empty to keep the term in English (e.g. brand names). Click **Зберегти**.

---

## Troubleshooting
| Problem | Fix |
|---|---|
| "Не знайдено звук" / sound bar doesn't move | Make sure Zoom is playing sound on this PC, and pick the device you actually hear it from. |
| Speech recognition yellow / "out of memory" | Close other heavy programs, then restart `START.bat`. |
| Public link says "Створюється…" for a long time | Check the internet connection or antivirus/firewall. The link changes on every launch, so re-post it. |
| Translation shows "⏳ повтор" | Gemini limit reached or no internet. It retries automatically; nothing is skipped. |
| Anything else | Send the newest file from the `logs` folder. |

To stop the app: close the black window, or click **⏻ Вимкнути програму**, or run `STOP.bat`.

---

## Files
| File / folder | What it is |
|---|---|
| `START.bat` | Launcher. Installs everything on first run, then starts the app. |
| `STOP.bat` | Stops the app. |
| `requirements*.txt` | Lists of components to install. |
| `config.example.json` | Example settings (the real `config.json` holds your keys; never share it). |
| `glossary.json` | Your term list. |
| `app/` | The program (sound capture, speech recognition, translation, web server, Zoom captions, public link, saving). |
| `web/viewer.html` | The page participants see. |
| `web/control.html` | The host's control panel. |
| `tests/` | Developer checks (not needed to run). |
| `.runtime/` | Auto-installed Python, packages and AI models (about 3 GB, created on first run). |
| `transcripts/` | Saved subtitles of every session. |
| `logs/` | Error logs. |

## Sharing with someone else
Zip only: `app`, `web`, `START.bat`, `STOP.bat`, `requirements.txt`, `requirements-gpu.txt`, `config.example.json`, `glossary.json`.
**Do not** include `.runtime`, `config.json`, `logs` or `transcripts`.
