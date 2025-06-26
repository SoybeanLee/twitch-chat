# twitch-chat

## Setup Python venv and install requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Follow the instruction [here](https://github.com/lay295/TwitchDownloader?tab=readme-ov-file#macos--getting-started) to install `TwitchDownloaderCLI` and `ffmpeg`

## Usage
```bash
python3 twitch_downloader.py -vod 2490741688
```