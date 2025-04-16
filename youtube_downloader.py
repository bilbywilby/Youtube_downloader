import PySimpleGUI as sg
import yt_dlp
import os
import threading
import json
import time
import hashlib
import uuid
import secrets
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import webbrowser
import logging
import logging.handlers
import platform
import subprocess
from queue import Queue, Empty
import validators
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import shutil
import psutil
import tempfile
from dataclasses import dataclass, asdict
from functools import lru_cache

# ----------- App Directories & Constants -----------

APP_ROOT = Path.home() / '.youtube_downloader'
CONFIG_PATH = APP_ROOT / 'config.json'
HISTORY_PATH = APP_ROOT / 'history.json'
QUEUE_PATH = APP_ROOT / 'queue.json'
LOG_DIR = APP_ROOT / 'logs'
DOWNLOADS_ROOT = APP_ROOT / 'downloads'
TEMP_DIR = APP_ROOT / 'temp'
LOG_FILE = LOG_DIR / 'downloader.log'
ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'webm', 'mp3', 'wav', 'm4a'}
MAX_FILENAME_LENGTH = 128

for d in [APP_ROOT, LOG_DIR, DOWNLOADS_ROOT, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------- Logging Setup -----------

rotating_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=1024*1024, backupCount=5
)
logging.basicConfig(
    handlers=[rotating_handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger("YouTubeDownloader")

# ----------- Utility & Security -----------

def sanitize_filename(filename: str) -> str:
    name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    return name[:MAX_FILENAME_LENGTH]

def is_safe_path(base_dir: str, user_path: str) -> bool:
    try:
        return Path(user_path).resolve().is_relative_to(Path(base_dir).resolve())
    except Exception:
        return False

def open_folder(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        sg.popup_error(f"Could not open downloads folder: {e}")

@dataclass
class AppConfig:
    download_dir: str = str(DOWNLOADS_ROOT)
    max_workers: int = 4
    default_format: str = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'
    default_resolution: str = 'best'
    default_audio_format: str = 'mp3'
    include_metadata: bool = True
    extract_thumbnail: bool = True
    theme: str = 'Reddit'
    use_ffmpeg: bool = False
    auto_queue: bool = False
    show_advanced: bool = False
    max_download_size: int = 2048
    rate_limit: int = 0
    history_enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 5
    min_disk_space: int = 1024
    preferred_protocol: str = 'https'

# ----------- Managers -----------

class ConfigManager:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        try:
            with open(CONFIG_PATH, 'r') as f:
                loaded = json.load(f)
                return AppConfig(**loaded)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Configuration file not found or invalid. Using default settings.")
            return AppConfig()
        except TypeError as e:
            logger.error(f"Type error loading config: {e}. Using default settings.")
            return AppConfig()

    def save_config(self):
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(asdict(self.config), f, indent=4)
        except IOError as e:
            logger.error(f"Error saving config: {e}")
            sg.popup_error(f"Error saving configuration: {e}")

class HistoryManager:
    def __init__(self):
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            with open(HISTORY_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def add_entry(self, entry: Dict[str, Any]):
        self.history.append(entry)
        self._save_history()

    def export_history(self, path: Path):
        try:
            with open(path, 'w') as f:
                json.dump(self.history, f, indent=4)
        except IOError as e:
            logger.error(f"Export history failed: {e}")
            sg.popup_error(f"Error exporting history: {e}")

    def clear_history(self):
        self.history = []
        self._save_history()

    def _save_history(self):
        try:
            with open(HISTORY_PATH, 'w') as f:
                json.dump(self.history, f, indent=4)
        except IOError as e:
            logger.error(f"Error saving history: {e}")
            sg.popup_error(f"Error saving download history: {e}")

class QueueManager:
    def __init__(self):
        self.queue: List[Dict[str, Any]] = self._load_queue()

    def _load_queue(self) -> List[Dict[str, Any]]:
        try:
            with open(QUEUE_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_queue(self, queue_items: List[Dict[str, Any]]):
        try:
            with open(QUEUE_PATH, 'w') as f:
                json.dump(queue_items, f, indent=4)
        except IOError as e:
            logger.error(f"Saving queue failed: {e}")
            sg.popup_error(f"Error saving download queue: {e}")

    def clear_queue(self):
        self.queue = []
        self.save_queue(self.queue)

class URLValidator:
    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
        except ValueError:
            return False
        youtube_patterns = [
            r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.*$',
            r'^(https?://)?(www\.)?youtube\.com/playlist\?.*$',
            r'^(https?://)?(www\.)?youtube\.com/watch\?.*$',
            r'^(https?://)?(www\.)?youtu\.be/.*$',
            r'^(https?://)?(www\.)?youtube\.com/shorts/.*$',
            r'^(https?://)?(www\.)?youtube\.com/channel/.*$',
            r'^(https?://)?(www\.)?youtube\.com/user/.*$'
        ]
        return any(re.match(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)

    @staticmethod
    def sanitize_url(url: str) -> str:
        return re.sub(r'[<>"\'\`;]', '', url.strip())

class FormatDetector:
    def __init__(self):
        self.ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': 'in_playlist'}
        self._format_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def detect_formats(self, url: str) -> Dict[str, List[Dict[str, Any]]]:
        if url in self._format_cache:
            return self._format_cache[url]
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = {'video': [], 'audio': []}
                for fmt in info.get('formats', []):
                    format_info = {
                        'format_id': fmt.get('format_id', ''),
                        'ext': fmt.get('ext', ''),
                        'filesize': fmt.get('filesize', 0),
                        'tbr': fmt.get('tbr', 0)
                    }
                    if fmt.get('vcodec', 'none') != 'none':
                        format_info.update({'height': fmt.get('height', 0), 'fps': fmt.get('fps', 0)})
                        formats['video'].append(format_info)
                    elif fmt.get('acodec', 'none') != 'none':
                        format_info.update({'abr': fmt.get('abr', 0)})
                        formats['audio'].append(format_info)
                self._format_cache[url] = formats
                return formats
        except Exception as e:
            logger.error(f"Format detection error for {url}: {e}")
            return {'video': [], 'audio': []}

class DownloadManager:
    def __init__(self, config: ConfigManager, window: sg.Window, history_manager: HistoryManager, queue_manager: QueueManager):
        self.config = config
        self.window = window
        self.download_queue: Queue[Dict[str, Any]] = Queue()
        self.format_detector = FormatDetector()
        self.history_manager = history_manager
        self.queue_manager = queue_manager
        self._active_downloads = set()
        self._cancel_event = threading.Event()

    def _check_disk_space(self, path: str) -> bool:
        try:
            free_space = psutil.disk_usage(path).free / (1024 * 1024)
            return free_space >= self.config.config.min_disk_space
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return False

    def _get_safe_output_path(self, filename: str, ext: str) -> str:
        safe_name = sanitize_filename(filename)
        today = datetime.now().strftime('%Y-%m-%d')
        dir_path = Path(self.config.config.download_dir) / today
        dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path / f"{safe_name}.{ext}")

    def add_to_queue(self, url: str, options: Dict[str, Any]) -> bool:
        sanitized_url = URLValidator.sanitize_url(url)
        if not URLValidator.is_valid_youtube_url(sanitized_url):
            sg.popup_error("Invalid YouTube URL")
            return False
        if not self._check_disk_space(self.config.config.download_dir):
            sg.popup_error(f"Insufficient disk space. Need at least {self.config.config.min_disk_space}MB free.")
            return False
        formats = self.format_detector.detect_formats(sanitized_url)
        if not formats['video'] and not formats['audio']:
            sg.popup_error("No valid formats found for this URL")
            return False
        queue_item = {
            'url': sanitized_url,
            'options': options,
            'formats': formats,
            'added_time': datetime.now().isoformat()
        }
        self.download_queue.put(queue_item)
        self._update_queue_display()
        return True

    def process_queue(self):
        with ThreadPoolExecutor(max_workers=self.config.config.max_workers) as executor:
            futures = []
            while not self.download_queue.empty():
                queue_item = self.download_queue.get()
                future = executor.submit(self._download_video, queue_item)
                self._active_downloads.add(future)
                futures.append(future)
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if self.config.config.history_enabled:
                        self.history_manager.add_entry(result)
                except Exception as e:
                    logger.error(f"Download failed: {e}")
                    self.window.write_event_value('-DOWNLOAD_ERROR-', str(e))
                finally:
                    self._active_downloads.remove(future)
        self.window.write_event_value('-QUEUE_FINISHED-', True)

    def _download_video(self, queue_item: Dict[str, Any]) -> Dict[str, Any]:
        url = queue_item['url']
        options = queue_item['options']
        ydl_opts = {
            'format': options['format'],
            'outtmpl': self._get_safe_output_path("%(title)s", options['format']),
            'writesubtitles': options.get('download_subs', False),
            'writeautomaticsub': options.get('download_subs', False),
            'extractaudio': options.get('extract_audio', False),
            'audioformat': options.get('audio_format', 'mp3'),
            'nocheckcertificate': False,
            'progress_hooks': [self._progress_hook],
            'ratelimit': self.config.config.rate_limit * 1024 if self.config.config.rate_limit > 0 else None,
            'max_filesize': self.config.config.max_download_size * 1024 * 1024 if self.config.config.max_download_size > 0 else None,
            'ffmpeg_location': shutil.which("ffmpeg") if self.config.config.use_ffmpeg else None,
            'protocol': self.config.config.preferred_protocol
        }
        for attempt in range(self.config.config.max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return {
                        'url': url,
                        'title': info.get('title', ''),
                        'filepath': ydl.prepare_filename(info),
                        'download_time': datetime.now().isoformat(),
                        'success': True
                    }
            except yt_dlp.DownloadError as e:
                if attempt == self.config.config.max_retries - 1:
                    logger.error(f"Download failed after {self.config.config.max_retries} attempts for {url}: {e}")
                    return {
                        'url': url,
                        'error': str(e),
                        'download_time': datetime.now().isoformat(),
                        'success': False
                    }
                time.sleep(self.config.config.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected download error: {e}")
                return {
                    'url': url,
                    'error': str(e),
                    'download_time': datetime.now().isoformat(),
                    'success': False
                }

    def _progress_hook(self, d):
        if d['status'] == 'downloading':
            progress = 0
            speed = d.get('speed', 0)
            eta = d.get('eta', 0)
            if d.get('total_bytes'):
                progress = int(d['downloaded_bytes'] / d['total_bytes'] * 100)
            elif d.get('total_bytes_estimate'):
                progress = int(d['downloaded_bytes'] / d['total_bytes_estimate'] * 100)
            self.window.write_event_value('-PROGRESS-', {
                'progress': progress,
                'speed': speed / 1024 if speed else 0,
                'eta': eta
            })
        elif d['status'] == 'finished':
            self.window.write_event_value('-FINISHED-', d['filename'])

    def _update_queue_display(self):
        queue_items = list(self.download_queue.queue)
        display_text = '\n'.join([
            f"{i+1}. {item['url']} ({item['options'].get('format', 'default')})"
            for i, item in enumerate(queue_items)
        ])
        self.window.write_event_value('-UPDATE_QUEUE-', display_text)

    def remove_from_queue(self):
        try:
            self.download_queue.get_nowait()
            self._update_queue_display()
            return True
        except Empty:
            sg.popup_error("Download queue is empty")
            return False

    def clear_queue(self):
        while not self.download_queue.empty():
            self.download_queue.get()
        self._update_queue_display()

# ----------- GUI Class -----------

class YoutubeDownloaderGUI:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.history_manager = HistoryManager()
        self.queue_manager = QueueManager()
        sg.theme(self.config_manager.config.theme)
        self.window = self._create_window()
        self.download_manager = DownloadManager(self.config_manager, self.window, self.history_manager, self.queue_manager)
        self.shortcuts = {
            'ctrl+q': '-EXIT-',
            'ctrl+a': '-ADD-',
            'ctrl+s': '-START-',
            'ctrl+d': '-DETECT-'
        }
        self.tray = sg.SystemTray(['', ['Show Window', 'Hide Window', '---', 'Exit']], filename=None)
        self._load_initial_queue()
        self._update_history_display()

    def _load_initial_queue(self):
        if self.config_manager.config.auto_queue:
            for item in self.queue_manager.queue:
                self.download_manager.download_queue.put(item)
            self.download_manager._update_queue_display()

    def _update_history_display(self):
        history_text = '\n'.join([
            f"{h.get('title', 'Unknown')} | {h.get('download_time', '')} | {'Success' if h.get('success') else 'Failed'}"
            for h in self.history_manager.history[-20:]
        ])
        self.window['-HISTORY-'].update(history_text)

    def _create_window(self) -> sg.Window:
        tooltips = {
            '-URL-': 'Enter a valid YouTube URL here',
            '-FORMAT-': 'Select the desired video format',
            '-QUALITY-': 'Select video quality',
            '-AUDIO_ONLY-': 'Extract audio only from the video',
            '-DIR-': 'Choose where to save downloaded files',
            '-SUBS-': 'Download video subtitles if available',
            '-AUDIO_FORMAT-': 'Choose the audio format for extraction'
        }
        url_section = [
            [sg.Text("Enter YouTube URL:", font=('Helvetica', 11))],
            [sg.Input(key='-URL-', size=(60, 1), font=('Helvetica', 10), tooltip=tooltips['-URL-'])],
            [sg.Button('🔍 Detect Format (Ctrl+D)', key='-DETECT-', size=(20, 1)),
             sg.Button('👁 Preview', key='-PREVIEW-', size=(15, 1))]
        ]
        options_section = [
            [sg.Frame('Download Options', [
                [sg.Text("Save to:", font=('Helvetica', 10))],
                [sg.Input(default_text=self.config_manager.config.download_dir,
                          key='-DIR-', size=(50, 1), tooltip=tooltips['-DIR-']),
                 sg.FolderBrowse(file_types=(("All Files", "*.*"),), tooltip="Choose download folder")],
                [sg.Frame('Video Options', [
                    [sg.Text("Format:"),
                     sg.Combo(['mp4', 'mkv', 'webm'], key='-FORMAT-', default_value='mp4', size=(10, 1), tooltip=tooltips['-FORMAT-'])],
                    [sg.Text("Quality:"),
                     sg.Combo(['best', '1080p', '720p', '480p', '360p'], key='-QUALITY-', default_value='best', size=(10, 1), tooltip=tooltips['-QUALITY-'])],
                    [sg.Checkbox('Download Subtitles', key='-SUBS-', tooltip=tooltips['-SUBS-'])]
                ])],
                [sg.Frame('Audio Options', [
                    [sg.Checkbox('Extract Audio', key='-AUDIO_ONLY-', tooltip=tooltips['-AUDIO_ONLY-'])],
                    [sg.Text("Format:"),
                     sg.Combo(['mp3', 'wav', 'm4a'], key='-AUDIO_FORMAT-', default_value='mp3', size=(10, 1), tooltip=tooltips['-AUDIO_FORMAT-'])]
                ])]
            ])]
        ]
        queue_section = [
            [sg.Frame('Download Queue', [
                [sg.Multiline(size=(70, 5), key='-QUEUE-', disabled=True, autoscroll=True)],
                [sg.Button('➕ Add to Queue (Ctrl+A)', key='-ADD-', size=(20, 1)),
                 sg.Button('➖ Remove', key='-REMOVE-', size=(15, 1)),
                 sg.Button('▶️ Start Download (Ctrl+S)', key='-START-', size=(20, 1)),
                 sg.Button('⚙️ Settings', key='-SETTINGS-', size=(15, 1))]
            ])]
        ]
        history_section = [
            [sg.Frame('Download History', [
                [sg.Multiline(size=(70, 5), key='-HISTORY-', disabled=True, autoscroll=True)],
                [sg.Button('Export History', key='-EXPORT_HISTORY-', size=(15, 1)),
                 sg.Button('Clear History', key='-CLEAR_HISTORY-', size=(15, 1))]
            ])]
        ]
        progress_section = [
            [sg.Frame('Progress', [
                [sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-')],
                [sg.Text('Status: ', font=('Helvetica', 10)), sg.Text('Waiting...', key='-STATUS-', font=('Helvetica', 10))],
                [sg.Text('Speed: ', font=('Helvetica', 10)), sg.Text('0 KB/s', key='-SPEED-', font=('Helvetica', 10))],
                [sg.Text('ETA: ', font=('Helvetica', 10)), sg.Text('--:--', key='-ETA-', font=('Helvetica', 10))]
            ])]
        ]
        layout = [
            url_section,
            options_section,
            queue_section,
            history_section,
            progress_section,
            [sg.Button('Open Downloads Folder', key='-OPEN_FOLDER-'), sg.Button('Exit (Ctrl+Q)', key='-EXIT-')]
        ]
        return sg.Window('YouTube Downloader', layout, finalize=True, return_keyboard_events=True, icon=None)

    def _handle_keyboard_shortcuts(self, event):
        for shortcut, action in self.shortcuts.items():
            ctrl_key = 'Control_L' in event or 'Control_R' in event
            key = event[-1].lower()
            if ctrl_key and key == shortcut.split('+')[1]:
                return action
        return event

    def _show_settings_window(self):
        layout = [
            [sg.Text("Download Settings", font=('Helvetica', 12, 'bold'))],
            [sg.Text("Max concurrent downloads:"), sg.Input(str(self.config_manager.config.max_workers), key='-MAX_WORKERS-', size=(5, 1))],
            [sg.Text("Download rate limit (KB/s, 0 for unlimited):"), sg.Input(str(self.config_manager.config.rate_limit), key='-RATE_LIMIT-', size=(10, 1))],
            [sg.Text("Max download size (MB):"), sg.Input(str(self.config_manager.config.max_download_size), key='-MAX_SIZE-', size=(10, 1))],
            [sg.Text("Maximum retry attempts:"), sg.Input(str(self.config_manager.config.max_retries), key='-MAX_RETRIES-', size=(5, 1))],
            [sg.Text("Retry delay (seconds):"), sg.Input(str(self.config_manager.config.retry_delay), key='-RETRY_DELAY-', size=(5, 1))],
            [sg.Checkbox('Enable download history', default=self.config_manager.config.history_enabled, key='-HISTORY-')],
            [sg.Checkbox('Use FFmpeg for post-processing', default=self.config_manager.config.use_ffmpeg, key='-USE_FFMPEG_SETTING-')],
            [sg.Checkbox('Auto Add to Queue', default=self.config_manager.config.auto_queue, key='-AUTO_QUEUE_SETTING-')],
            [sg.Text("Preferred Protocol:"), sg.Radio('HTTPS', "PROTOCOL", default=self.config_manager.config.preferred_protocol == 'https', key='-HTTPS_PROTO-'), sg.Radio('HTTP', "PROTOCOL", default=self.config_manager.config.preferred_protocol == 'http', key='-HTTP_PROTO-')],
            [sg.Text("Theme:"), sg.Combo(sg.theme_list(), default_value=self.config_manager.config.theme, key='-THEME-', size=(20, 1))],
            [sg.Text("Minimum disk space (MB):"), sg.Input(str(self.config_manager.config.min_disk_space), key='-MIN_DISK_SPACE-', size=(10, 1))],
            [sg.Button('Save'), sg.Button('Cancel')]
        ]
        settings_window = sg.Window('Settings', layout, modal=True)
        event, values = settings_window.read(close=True)
        if event == 'Save':
            try:
                self.config_manager.config = AppConfig(
                    download_dir=values['-DIR-'],
                    max_workers=int(values['-MAX_WORKERS-']),
                    rate_limit=int(values['-RATE_LIMIT-']),
                    max_download_size=int(values['-MAX_SIZE-']),
                    max_retries=int(values['-MAX_RETRIES-']),
                    retry_delay=int(values['-RETRY_DELAY-']),
                    history_enabled=values['-HISTORY-'],
                    theme=values['-THEME-'],
                    use_ffmpeg=values['-USE_FFMPEG_SETTING-'],
                    auto_queue=values['-AUTO_QUEUE_SETTING-'],
                    min_disk_space=int(values['-MIN_DISK_SPACE-']),
                    preferred_protocol='https' if values['-HTTPS_PROTO-'] else 'http'
                )
                self.config_manager.save_config()
                sg.theme(self.config_manager.config.theme)
                self.window.close()
                self.window = self._create_window()
                self.download_manager = DownloadManager(self.config_manager, self.window, self.history_manager, self.queue_manager)
            except ValueError as e:
                sg.popup_error("Invalid input values")

    def _show_about_dialog(self):
        sg.popup("YouTube Downloader", "Version 1.0", "Created with PySimpleGUI and yt-dlp", "© 2025 Your Name/Organization", grab_anywhere=True)

    def run(self):
        while True:
            tray_event = self.tray.read(timeout=100)
            if tray_event == 'Exit':
                break
            elif tray_event in ('Show Window', 'Double Click'):
                self.window.un_hide()
            elif tray_event == 'Hide Window':
                self.window.hide()
            event, values = self.window.read(timeout=100)
            if event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT:
                if sg.popup_yes_no('Do you want to minimize to tray instead of closing?', title='Minimize to Tray') == 'Yes':
                    self.window.hide()
                    continue
                else:
                    break
            event = self._handle_keyboard_shortcuts(event)
            if event in (None, '-EXIT-'):
                break
            elif event == '-OPEN_FOLDER-':
                open_folder(self.config_manager.config.download_dir)
            elif event == '-EXPORT_HISTORY-':
                filepath = sg.popup_get_file('Save download history to:', save_as=True, file_types=(("JSON files", "*.json"),))
                if filepath:
                    self.history_manager.export_history(Path(filepath))
            elif event == '-CLEAR_HISTORY-':
                self.history_manager.clear_history()
                self._update_history_display()
            elif event == '-SETTINGS-':
                self._show_settings_window()
            elif event == '-DETECT-':
                url = URLValidator.sanitize_url(values['-URL-'])
                if URLValidator.is_valid_youtube_url(url):
                    formats = self.download_manager.format_detector.detect_formats(url)
                    format_info = "Available Formats:\n"
                    if formats['video']:
                        format_info += "\nVideo:\n"
                        for fmt in formats['video']:
                            format_info += f"- {fmt['height']}p ({fmt['ext']}, {fmt['tbr']}k)\n"
                    if formats['audio']:
                        format_info += "\nAudio:\n"
                        for fmt in formats['audio']:
                            format_info += f"- {fmt['ext']} ({fmt['abr']}k)\n"
                    sg.popup_scrolled(format_info, title='Available Formats')
                else:
                    sg.popup_error("Invalid YouTube URL")
            elif event == '-PREVIEW-':
                url = URLValidator.sanitize_url(values['-URL-'])
                if URLValidator.is_valid_youtube_url(url):
                    webbrowser.open_new_tab(url)
                else:
                    sg.popup_error("Invalid YouTube URL")
            elif event == '-ADD-':
                url = URLValidator.sanitize_url(values['-URL-'])
                if URLValidator.is_valid_youtube_url(url):
                    options = {
                        'download_dir': values['-DIR-'],
                        'format': values['-FORMAT-'],
                        'quality': values['-QUALITY-'],
                        'download_subs': values['-SUBS-'],
                        'extract_audio': values['-AUDIO_ONLY-'],
                        'audio_format': values['-AUDIO_FORMAT-']
                    }
                    self.download_manager.add_to_queue(url, options)
                else:
                    sg.popup_error("Invalid YouTube URL")
            elif event == '-REMOVE-':
                self.download_manager.remove_from_queue()
            elif event == '-START-':
                if not self.download_manager.download_queue.empty():
                    self.window['-STATUS-'].update('Downloading...')
                    threading.Thread(target=self.download_manager.process_queue, daemon=True).start()
                    self.window['-START-'].update(disabled=True)
                else:
                    sg.popup_error("Download queue is empty")
            elif event == '-PROGRESS-':
                progress_info = values[event]
                self.window['-PROGRESS-'].update(progress_info['progress'])
                self.window['-SPEED-'].update(f"{progress_info['speed']:.1f} KB/s")
                if progress_info['eta']:
                    eta_mins = progress_info['eta'] // 60
                    eta_secs = progress_info['eta'] % 60
                    self.window['-ETA-'].update(f"{eta_mins:02d}:{eta_secs:02d}")
            elif event == '-FINISHED-':
                self.window['-STATUS-'].update('Download Complete!')
                sg.popup_notify('Download Complete!', values[event])
            elif event == '-DOWNLOAD_ERROR-':
                self.window['-STATUS-'].update('Error!')
                sg.popup_error(f"Download Error: {values[event]}")
            elif event == '-UPDATE_QUEUE-':
                self.window['-QUEUE-'].update(values[event])
            elif event == '-QUEUE_FINISHED-':
                self.window['-STATUS-'].update('Queue Finished!')
                self.window['-START-'].update(disabled=False)
                self.window['-SPEED-'].update('0 KB/s')
                self.window['-ETA-'].update('--:--')
                self._update_history_display()
        self._cleanup()
        self.window.close()
        self.tray.Stop()

    def _cleanup(self):
        try:
            if self.config_manager.config.auto_queue:
                self.queue_manager.save_queue(list(self.download_manager.download_queue.queue))
            self.config_manager.save_config()
            for file in TEMP_DIR.glob('*'):
                try:
                    if file.is_file():
                        file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {file}: {e}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == '__main__':
    try:
        app = YoutubeDownloaderGUI()
        app.run()
    except Exception as e:
        logger.critical(f"Application crash: {e}", exc_info=True)
        sg.popup_error(f"Critical error: {e}\nCheck logs for details.")