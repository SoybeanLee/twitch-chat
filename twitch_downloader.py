import sys, os, math
import argparse
import subprocess
from bs4 import BeautifulSoup
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from faster_whisper import WhisperModel


class TwitchDownloader:
    def __init__(self, video_id, output_dir):
        self.video_id = video_id
        self.video_url = f'https://www.twitch.tv/videos/{video_id}'
        self.output_dir = f'{output_dir}/{video_id}'
        self.audio_m4a_path = f'{self.output_dir}/{video_id}.m4a'
        self.wav_path = f'{self.output_dir}/{video_id}.wav'
        self.transcript_tsv_path = f'{self.output_dir}/{video_id}_transcript.tsv'
        self.chat_html_path = f'{self.output_dir}/{video_id}_chat.html'
        self.chat_tsv_path = f'{self.output_dir}/{video_id}_chat.tsv'
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_chat(self):
        print(f'\n>>> Downloading chat for video ID: {self.video_id}')

        # check if chat HTML file already exists
        if os.path.exists(self.chat_html_path):
            print(f'chat HTML file already exists: {self.chat_html_path}')
            print('you must delete it manually if you want to re-download the chat.')
            sys.exit(1)

        # invoke TwitchDownloaderCLI executable to download the chat
        # !./TwitchDownloaderCLI chatdownload --id {VOD_ID} -o "{CHAT_HTML}"
        subprocess.run(['./TwitchDownloaderCLI', 'chatdownload', 
                        '--id', self.video_id, 
                        '-o', self.chat_html_path])
        print(f"Chat downloaded successfully: {self.chat_html_path}")

        with open(self.chat_html_path, 'r', encoding='utf-8') as fin:
            soup = BeautifulSoup(fin, 'html.parser')

        data = []
        for pre in soup.find_all("pre", class_="comment-root"):
            time_at_second = pre.text.split("]")[0].strip("[")
            auth = pre.find("span", class_="comment-author")
            author = auth.text.strip() if auth else ""
            msg_tag = pre.find("span", class_="comment-message")
            msg = msg_tag.text.strip(": ") if msg_tag else ""
            data.append({'time': time_at_second, 'author': author, 'message': msg})
        
        # Convert to DataFrame and save to TSV
        if not data:
            print('No chat data found.')
        else:
            print(f'Found {len(data):,} chat messages.')
            chat_df = pd.DataFrame(data)
            # save DataFrame to CSV
            chat_df.to_csv(f'{self.chat_tsv_path}', index=False, sep='\t')
            print(f'Chat data saved to {self.chat_tsv_path}')

    def fetch_audio(self):
        print(f'\n>>> Downloading audio for video ID: {self.video_id}')

        # check if audio wav file already exists
        if os.path.exists(self.wav_path):
            print(f'audio wav file already exists: {self.wav_path}')
            print('you must delete it manually if you want to re-download audio wav.')
            sys.exit(1)

        # invoke yt-dlp to download the audio
        # !yt-dlp --extract-audio --audio-format m4a -o "{AUDIO_M4A}" "{VOD_URL}"
        subprocess.run(['yt-dlp', '--extract-audio', 
                        '--audio-format', 'm4a', 
                        '-o', self.audio_m4a_path, 
                        '--ffmpeg-location', './ffmpeg',
                        self.video_url])
        print(f"m4a file downloaded successfully: {self.audio_m4a_path}")

        subprocess.run(['./ffmpeg', '-y', '-i', self.audio_m4a_path, '-ar', '16000', '-ac', '1', self.wav_path])
        print(f"wav file converted successfully: {self.wav_path}")
    
    def transcribe_audio(self):
        def _format_ts(t):
            h, rem = divmod(t, 3600)
            m, s   = divmod(rem, 60)
            return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

        print(f'\n>>> Transcribing audio for video ID: {self.video_id}')

        # check if transcript file already exists
        if os.path.exists(self.transcript_tsv_path):
            print(f'transcript file already exists: {self.transcript_tsv_path}')
            print('you must delete it manually if you want to re-transcribe the audio.')
            sys.exit(1)


        sf_audio, sr = sf.read(self.wav_path, dtype='float32')
        whisper_model = WhisperModel('tiny', device='cpu', compute_type='int8')

        chunk_sec  = 120
        chunk_size = int(chunk_sec * sr)
        num_chunk = math.ceil(len(sf_audio) / chunk_size)

        all_segments = []

        for idx in tqdm(range(num_chunk), desc='Transcribing audio', unit='chunk'):
            start = idx * chunk_size
            end   = min(start + chunk_size, len(sf_audio))
            chunk = sf_audio[start:end]

            segments, _ = whisper_model.transcribe(
                chunk,
                beam_size=1,
                language='en'
            )

            offset = start / sr
            for seg in segments:
                all_segments.append({
                    'start': seg.start + offset,
                    'end':   seg.end   + offset,
                    'text':  seg.text
                })
        
        if not all_segments:
            print('No transcript data found.')
        else:
            print(f'Found {len(all_segments):,} all_segments of transcript.')

            transcript_df = pd.DataFrame(all_segments)
            transcript_df = transcript_df.sort_values('start').reset_index(drop=True)

            transcript_df['time'] = transcript_df['start'].apply(_format_ts)
            transcript_df = transcript_df[['time', 'text']]
            transcript_df.to_csv(f'{self.transcript_tsv_path}', index=False, sep='\t')
            print(f'Transcript saved to {self.transcript_tsv_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Twitch VODs.')
    parser.add_argument('-vod', 
                        '--video_id', 
                        required=True, 
                        help='Twitch video ID to download')
    parser.add_argument('-output', 
                        '--output_dir', 
                        required=False, 
                        default='data', 
                        help='Directory to save the downloaded video')
    args = parser.parse_args()
    video_id = args.video_id
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    downloader = TwitchDownloader(video_id, output_dir)
    downloader.fetch_chat()
    downloader.fetch_audio()
    downloader.transcribe_audio()
