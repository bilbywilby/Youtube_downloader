import PySimpleGUI as sg
import yt_dlp
import os
import re
import logging
import logging.handlers
import threading
import time
import uuid
import webbrowser
import shutil
import psutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import validators
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse, parse_qs
from functools import lru_cache, partial

# ----------- Version Info -----------
VERSION = "2.0.0"
AUTHOR = "bilbywilby"
UPDATED = "2025-04-16"

# ----------- App Directories & Constants -----------
APP_ROOT = Path.home() / '.youtube_downloader'
CONFIG_PATH = APP_ROOT / 'config.json'
HISTORY_PATH = APP_ROOT / 'history.json'
QUEUE_PATH = APP_ROOT / 'queue.json'
LOG_DIR = APP_ROOT / 'logs'
DOWNLOADS_ROOT = APP_ROOT / 'downloads'
TEMP_DIR = APP_ROOT / 'temp'
CACHE_DIR = APP_ROOT / 'cache'
LOG_FILE = LOG_DIR / 'downloader.log'

# File formats and limits
ALLOWED_VIDEO_FORMATS = {'mp4', 'mkv', 'webm'}
ALLOWED_AUDIO_FORMATS = {'mp3', 'wav', 'm4a', 'aac', 'flac'}
MAX_FILENAME_LENGTH = 128
MIN_DISK_SPACE_MB = 1024  # 1GB minimum
MAX_RETRIES = 3
RETRY_DELAY = 5
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for downloads

# ----------- Logging Setup -----------
def setup_logging():
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("YouTubeDownloader")


logger = setup_logging()

# Create necessary directories
for d in [APP_ROOT, LOG_DIR, DOWNLOADS_ROOT, TEMP_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {d}")

logger = setup_logging()

# ----------- Error Classes -----------


class DownloaderError(Exception):
    """
    DownloaderError is a custom exception class used as the base for all downloader-related errors.

    Attributes:
        message (str): A descriptive error message explaining the cause of the exception.

    Methods:
        __init__(message: str): Initializes the DownloaderError with a specific error message.
    """
    """Base exception for downloader errors"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ValidationError(DownloaderError):
    """Validation related errors"""
    pass

class DiskSpaceError(DownloaderError):
    """Disk space related errors"""
    pass

class NetworkError(DownloaderError):
    """Network related errors"""
    pass

class ConfigError(DownloaderError):
    """Configuration related errors"""
    pass

# ----------- Data Classes -----------
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
    resume_downloads: bool = True
    verify_ssl: bool = True
    proxy: str = ''
    download_archive: bool = True
    archive_path: str = str(APP_ROOT / 'downloaded.txt')
    notify_on_complete: bool = True

    def validate(self) -> bool:
        """Validate configuration values"""
        try:
            self.max_workers = max(1, min(int(self.max_workers), 10))
            self.rate_limit = max(0, int(self.rate_limit))
            self.max_download_size = max(0, int(self.max_download_size))
            self.max_retries = max(1, min(int(self.max_retries), 10))
            self.retry_delay = max(1, min(int(self.retry_delay), 30))
            self.min_disk_space = max(100, int(self.min_disk_space))

            # Validate download directory
            download_dir = Path(self.download_dir)
            if not download_dir.exists():
                download_dir.mkdir(parents=True, exist_ok=True)

            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

@dataclass
class DownloadItem:
    url: str
    title: Optional[str] = None
    format: str = 'mp4'
    quality: str = 'best'
    download_path: Optional[str] = None
    status: str = 'pending'
    progress: int = 0
    speed: float = 0
    eta: int = 0
    error: Optional[str] = None
    video_id: Optional[str] = None
    filesize: Optional[int] = None
    download_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DownloadItem':
        return cls(**data)
# ----------- Utility Classes -----------
class URLValidator:
    """Enhanced URL validation and processing"""

    YOUTUBE_PATTERNS = [
        r'^https?:\/\/(?:www\.)?youtube\.com\/watch\?v=[\w-]+',
        r'^https?:\/\/(?:www\.)?youtube\.com\/shorts\/[\w-]+',
        r'^https?:\/\/youtu\.be\/[\w-]+',
        r'^https?:\/\/(?:www\.)?youtube\.com\/playlist\?list=[\w-]+',
    ]

    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False

        try:
            return any(re.match(pattern, url, re.IGNORECASE)
                      for pattern in URLValidator.YOUTUBE_PATTERNS)
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False

@staticmethod
@lru_cache(maxsize=128)
def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL with caching"""
    try:
        if not url or not isinstance(url, str):
            return None

        # Handle youtu.be URLs
        if 'youtu.be' in url:
            video_id = url.split('/')[-1].split('?')[0]
            return video_id if video_id else None

        # Handle youtube.com URLs
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc:
            if '/watch' in url:
                video_id = parse_qs(parsed_url.query).get('v', [None])[0]
            elif '/shorts/' in url:
                video_id = url.split('/shorts/')[1].split('?')[0]
            elif '/live/' in url:
                video_id = url.split('/live/')[1].split('?')[0]
            else:
                return None

            # Validate video ID format
            if video_id and re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
                return video_id

        return None
    except (ValueError, AttributeError, IndexError) as e:
        logger.error(f"Error extracting video ID: {e}")
        return None

    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL by removing potentially harmful characters"""
        try:
            if not url or not isinstance(url, str):
                return ''

            # Remove whitespace and common unsafe characters
            sanitized = re.sub(r'[<>"\'`;)(]', '', url.strip())

            # Ensure proper URL encoding
            return urllib.parse.quote(sanitized, safe=':/?=&')
        except Exception as e:
            logger.error(f"Error sanitizing URL: {e}")
            return ''

class FileManager:
    """Handle file operations with safety checks"""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Create safe filename"""
        # Remove invalid characters
        safe_name = re.sub(r'[<>:"/|?*]', '', filename)
        # Replace spaces and multiple dots
        safe_name = re.sub(r's+', '_', safe_name)
        safe_name = re.sub(r'.+', '.', safe_name)
        # Limit length
        return safe_name[:MAX_FILENAME_LENGTH]

    @staticmethod
    def get_safe_path(base_dir: Union[str, Path], filename: str) -> Path:
        """Generate safe file path with date-based organization"""
        base_path = Path(base_dir)
        date_dir = base_path / datetime.now().strftime('%Y-%m')
        date_dir.mkdir(parents=True, exist_ok=True)

        safe_name = FileManager.sanitize_filename(filename)
        path = date_dir / safe_name

        # Handle duplicates
        counter = 1
        stem = path.stem
        suffix = path.suffix
        while path.exists():
            path = date_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        return path

    @staticmethod
    def check_disk_space(path: Union[str, Path], required_mb: int) -> bool:
        """Check if there's enough disk space"""
        try:
            free_space = psutil.disk_usage(str(path)).free / (1024 * 1024)  # MB
            return free_space >= required_mb
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return False

    @staticmethod
    def create_temp_file() -> Path:
        """Create a temporary file for downloads"""
        return Path(tempfile.mktemp(dir=TEMP_DIR, suffix='.part'))

    @staticmethod
    def cleanup_temp_files():
        """Clean up temporary files"""
        try:
            for file in TEMP_DIR.glob('*.part'):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")
        except Exception as e:
            logger.error(f"Error during temp cleanup: {e}")

class SystemManager:
    """System-related operations"""

    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is available"""
        return bool(shutil.which('ffmpeg'))

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information"""
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': {str(DOWNLOADS_ROOT): psutil.disk_usage(str(DOWNLOADS_ROOT))._asdict()}
        }

    @staticmethod
    def open_folder(path: Union[str, Path]):
        """Open folder in system file explorer"""
        try:
            path = str(Path(path).resolve())
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", path], check=True)
            else:
                subprocess.run(["xdg-open", path], check=True)
        except Exception as e:
            logger.error(f"Failed to open folder {path}: {e}")
            raise OSError(f"Could not open folder: {e}")

class FormatDetector:
    """Enhanced format detection with caching"""

    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist'
        }
        self._format_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    @lru_cache(maxsize=100)
    def detect_formats(self, url: str) -> Dict[str, List[Dict[str, Any]]]:
        """Detect available formats with caching"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = {'video': [], 'audio': []}

                if not info:
                    raise ValueError("No format information found")

                for fmt in info.get('formats', []):
                    format_info = {
                        'format_id': fmt.get('format_id', ''),
                        'ext': fmt.get('ext', ''),
                        'filesize': fmt.get('filesize', 0),
                        'tbr': fmt.get('tbr', 0),
                        'vcodec': fmt.get('vcodec', 'none'),
                        'acodec': fmt.get('acodec', 'none'),
                    }

                    if fmt.get('vcodec', 'none') != 'none':
                        format_info.update({
                            'height': fmt.get('height', 0),
                            'width': fmt.get('width', 0),
                            'fps': fmt.get('fps', 0),
                            'dynamic_range': fmt.get('dynamic_range', 'SDR'),
                        })
                        formats['video'].append(format_info)
                    elif fmt.get('acodec', 'none') != 'none':
                        format_info.update({
                            'abr': fmt.get('abr', 0),
                            'asr': fmt.get('asr', 0),
                        })
                        formats['audio'].append(format_info)

                # Sort formats by quality
                formats['video'].sort(key=lambda x: (x.get('height', 0), x.get('tbr', 0)), reverse=True)
                formats['audio'].sort(key=lambda x: x.get('abr', 0), reverse=True)

                return formats

        except Exception as e:
            logger.error(f"Format detection error for {url}: {e}")
            return {'video': [], 'audio': []}

    def clear_cache(self):
        """Clear format detection cache"""
        self.detect_formats.cache_clear()

# ----------- Download Management -----------
class DownloadManager:
    """Enhanced download manager with retry and resume capabilities"""

    def __init__(self, config_manager, window: sg.Window, history_manager, queue_manager):
        self.config = config_manager
        self.window = window
        self.download_queue: Queue[DownloadItem] = Queue()
        self.format_detector = FormatDetector()
        self.history_manager = history_manager
        self.queue_manager = queue_manager
        self._active_downloads = set()
        self._cancel_event = threading.Event()

    def add_to_queue(self, url: str, options: Dict[str, Any]) -> bool:
        """Add a new download to the queue with validation"""
        try:
            # Validate URL
            sanitized_url = URLValidator.sanitize_url(url)
            if not URLValidator.is_valid_youtube_url(sanitized_url):
                raise ValidationError("Invalid YouTube URL")

            # Check disk space
            if not FileManager.check_disk_space(
                self.config.config.download_dir,
                self.config.config.min_disk_space
            ):
                raise DiskSpaceError(
                    f"Insufficient disk space. Need at least {self.config.config.min_disk_space}MB free."
                )

            # Detect formats
            formats = self.format_detector.detect_formats(sanitized_url)
            if not formats['video'] and not formats['audio']:
                raise ValidationError("No valid formats found for this URL")

            # Create download item
            download_item = DownloadItem(
                url=sanitized_url,
                format=options.get('format', 'mp4'),
                quality=options.get('quality', 'best'),
                download_path=options.get('download_dir'),
                video_id=URLValidator.extract_video_id(sanitized_url)
            )

            self.download_queue.put(download_item)
            self._update_queue_display()
            return True

        except Exception as e:
            logger.error(f"Failed to add URL to queue: {e}")
            sg.popup_error(str(e))
            return False

    def process_queue(self):
        """Process the download queue with improved error handling"""
        try:
            with ThreadPoolExecutor(max_workers=self.config.config.max_workers) as executor:
                futures = []
                while not self.download_queue.empty() and not self._cancel_event.is_set():
                    download_item = self.download_queue.get()
                    future = executor.submit(self._download_video, download_item)
                    self._active_downloads.add(future)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if self.config.config.history_enabled and result:
                            self.history_manager.add_entry(result.to_dict())
                    except Exception as e:
                        logger.error(f"Download failed: {e}")
                        self.window.write_event_value('-DOWNLOAD_ERROR-', str(e))
                    finally:
                        self._active_downloads.remove(future)

        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        finally:
            self.window.write_event_value('-QUEUE_FINISHED-', True)

    def _download_video(self, download_item: DownloadItem) -> Optional[DownloadItem]:
        """Download a single video with resume capability"""
        try:
            # Prepare download options
            ydl_opts = self._prepare_download_options(download_item)

            for attempt in range(self.config.config.max_retries):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        # Extract video information
                        info = ydl.extract_info(download_item.url, download=False)
                        if not info:
                            raise ValueError("No video information found")

                        # Update download item with video information
                        download_item.title = info.get('title', '')
                        download_item.filesize = info.get('filesize', 0)

                        # Perform the download
                        ydl.download([download_item.url])

                        # Update download item with success information
                        download_item.status = 'completed'
                        download_item.download_time = datetime.now().isoformat()
                        return download_item

                except yt_dlp.DownloadError as e:
                    if attempt == self.config.config.max_retries - 1:
                        raise
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    time.sleep(self.config.config.retry_delay)

            raise yt_dlp.DownloadError(
                f"Failed after {self.config.config.max_retries} attempts"
            )

        except Exception as e:
            logger.error(f"Download failed for {download_item.url}: {e}")
            download_item.status = 'failed'
            download_item.error = str(e)
            return download_item

    def _prepare_download_options(self, download_item: DownloadItem) -> Dict[str, Any]:
        """Prepare yt-dlp options for download"""
        output_template = self._get_output_template(download_item)

        opts = {
            'format': self._get_format_string(download_item.quality, download_item.format),
            'outtmpl': output_template,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'srt',
            'keepvideo': True,
            'writedescription': True,
            'writethumbnail': True,
            'writeannotations': True,
            'writeinfojson': True,
            'concurrent_fragment_downloads': self.config.config.max_workers,
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': True,
            'progress_hooks': [partial(self._progress_hook, download_item)],
            'postprocessors': self._get_postprocessors(download_item)
        }

        # Add rate limiting if configured
        if self.config.config.rate_limit > 0:
            opts['ratelimit'] = self.config.config.rate_limit * 1024

        # Add proxy if configured
        if self.config.config.proxy:
            opts['proxy'] = self.config.config.proxy

        # Add ffmpeg options if available
        if self.config.config.use_ffmpeg and SystemManager.check_ffmpeg():
            opts['ffmpeg_location'] = shutil.which('ffmpeg')
            opts['prefer_ffmpeg'] = True

        return opts

    def _get_format_string(self, quality: str, format: str) -> str:
        """Generate format string based on quality and format"""
        if quality == 'best':
            return f'bestvideo[ext={format}]+bestaudio[ext=m4a]/best[ext={format}]/best'

        height = quality.replace('p', '')
        return f'bestvideo[height<={height}][ext={format}]+bestaudio[ext=m4a]/best[height<={height}][ext={format}]/best'

    def _get_output_template(self, download_item: DownloadItem) -> str:
        """Generate output template with organized structure"""
        base_dir = Path(download_item.download_path or self.config.config.download_dir)
        date_str = datetime.now().strftime('%Y-%m')
        return str(base_dir / date_str / '%(title)s.%(ext)s')

    def _get_postprocessors(self, download_item: DownloadItem) -> List[Dict[str, Any]]:
        """Configure post-processors for the download"""
        postprocessors = []

        # Embed metadata
        postprocessors.append({
            'key': 'FFmpegMetadata',
            'add_metadata': True,
        })

        # Embed thumbnail if possible
        if self.config.config.extract_thumbnail:
            postprocessors.append({
                'key': 'EmbedThumbnail',
                'already_have_thumbnail': False,
            })

        # Extract audio if requested
        if download_item.format in ALLOWED_AUDIO_FORMATS:
            postprocessors.append({
                'key': 'FFmpegExtractAudio',
                'preferredcodec': download_item.format,
                'preferredquality': '192',
            })

        return postprocessors

    def _progress_hook(self, download_item: DownloadItem, d: Dict[str, Any]):
        """Handle download progress updates"""
        if d['status'] == 'downloading':
            # Calculate progress
            progress = 0
            speed = d.get('speed', 0)
            eta = d.get('eta', 0)

            if d.get('total_bytes'):
                progress = int(d['downloaded_bytes'] / d['total_bytes'] * 100)
            elif d.get('total_bytes_estimate'):
                progress = int(d['downloaded_bytes'] / d['total_bytes_estimate'] * 100)

            # Update download item
            download_item.progress = progress
            download_item.speed = speed / 1024 if speed else 0
            download_item.eta = eta

            # Update GUI
            self.window.write_event_value('-PROGRESS-', {
                'progress': progress,
                'speed': speed / 1024 if speed else 0,
                'eta': eta,
                'url': download_item.url
            })

        elif d['status'] == 'finished':
            self.window.write_event_value('-FINISHED-', {
                'filename': d['filename'],
                'url': download_item.url
            })

    def _update_queue_display(self):
        """Update the queue display in the GUI"""
        try:
            queue_items = list(self.download_queue.queue)
            display_text = 'n'.join([
                f"{i+1}. {item.url} ({item.format}, {item.quality})"
                for i, item in enumerate(queue_items)
            ])
            self.window.write_event_value('-UPDATE_QUEUE-', display_text)
        except Exception as e:
            logger.error(f"Failed to update queue display: {e}")

    def cancel_all_downloads(self):
        """Cancel all active downloads"""
        self._cancel_event.set()
        for future in self._active_downloads:
            future.cancel()
        self.clear_queue()

    def clear_queue(self):
        """Clear the download queue"""
        while not self.download_queue.empty():
            try:
                self.download_queue.get_nowait()
            except Empty:
                break
        self._update_queue_display()

# ----------- GUI Implementation -----------

class YoutubeDownloaderGUI:
    """Main GUI implementation with enhanced features"""

    def __init__(self):
        from config_manager import ConfigManager
        from history_manager import HistoryManager
        try:
            from queue_manager import QueueManager
        except ImportError:
            raise ImportError("The 'queue_manager' module is missing. Ensure 'queue_manager.py' exists in the same directory or is in the Python path.")

        self.config_manager = ConfigManager()
        self.history_manager = HistoryManager()
        self.queue_manager = QueueManager()

        # Set theme
        sg.theme(self.config_manager.config.theme)

        # Create window
        self.window = self._create_window()
        self.download_manager = DownloadManager(
            self.config_manager,
            self.window,
            self.history_manager,
            self.queue_manager
        )

        # Set up keyboard shortcuts
        self.shortcuts = {
            'ctrl+q': '-EXIT-',
            'ctrl+a': '-ADD-',
            'ctrl+s': '-START-',
            'ctrl+d': '-DETECT-',
            'ctrl+p': '-PREVIEW-',
            'ctrl+c': '-CANCEL-',
            'ctrl+o': '-OPEN_FOLDER-'
        }

        # Create system tray
        self.tray = self._create_tray()

        # Initialize
        self._load_initial_queue()
        self._update_history_display()

        logger.info("Application initialized successfully")

    def _create_window(self) -> sg.Window:
        """Create main application window with improved layout"""
        # Tooltips for UI elements
        tooltips = {
            '-URL-': 'Enter a valid YouTube URL here',
            '-FORMAT-': 'Select the desired video format',
            '-QUALITY-': 'Select video quality',
            '-AUDIO_ONLY-': 'Extract audio only from the video',
            '-DIR-': 'Choose where to save downloaded files',
            '-SUBS-': 'Download video subtitles if available',
            '-AUDIO_FORMAT-': 'Choose the audio format for extraction',
            '-PLAYLIST-': 'Download entire playlist'
        }

        # URL Section
        url_section = [
            [sg.Text("YouTube URL:", font=('Helvetica', 11), pad=((5,5),(10,5)))],
            [sg.Input(key='-URL-', size=(60, 1), font=('Helvetica', 10),
                     tooltip=tooltips['-URL-'], pad=((5,5),(0,5)))],
            [sg.Button('🔍 Detect Format', key='-DETECT-', size=(15, 1)),
             sg.Button('👁 Preview', key='-PREVIEW-', size=(15, 1)),
             sg.Button('📋 Paste', key='-PASTE-', size=(15, 1))]
        ]

        # Options Section
        options_section = [
            [sg.Frame('Download Options', [
                [sg.Text("Save to:", font=('Helvetica', 10))],
                [sg.Input(default_text=self.config_manager.config.download_dir,
                         key='-DIR-', size=(50, 1), tooltip=tooltips['-DIR-']),
                 sg.FolderBrowse()],
                [sg.Frame('Video Options', [
                    [sg.Text("Format:"),
                     sg.Combo(list(ALLOWED_VIDEO_FORMATS), key='-FORMAT-',
                             default_value='mp4', size=(10, 1),
                             tooltip=tooltips['-FORMAT-'])],
                    [sg.Text("Quality:"),
                     sg.Combo(['best', '4K', '1440p', '1080p', '720p', '480p', '360p'],
                             key='-QUALITY-', default_value='1080p',
                             size=(10, 1), tooltip=tooltips['-QUALITY-'])],
                    [sg.Checkbox('Download Subtitles', key='-SUBS-',
                               tooltip=tooltips['-SUBS-'])],
                    [sg.Checkbox('Download Playlist', key='-PLAYLIST-',
                               tooltip=tooltips['-PLAYLIST-'])]
                ])],
                [sg.Frame('Audio Options', [
                    [sg.Checkbox('Extract Audio Only', key='-AUDIO_ONLY-',
                               tooltip=tooltips['-AUDIO_ONLY-'])],
                    [sg.Text("Format:"),
                     sg.Combo(list(ALLOWED_AUDIO_FORMATS), key='-AUDIO_FORMAT-',
                             default_value='mp3', size=(10, 1),
                             tooltip=tooltips['-AUDIO_FORMAT-'])]
                ])]
            ])]
        ]

        # Queue Section
        queue_section = [
            [sg.Frame('Download Queue', [
                [sg.Multiline(size=(70, 5), key='-QUEUE-', disabled=True,
                            autoscroll=True, font=('Courier', 10))],
                [sg.Button('➕ Add to Queue', key='-ADD-', size=(15, 1)),
                 sg.Button('➖ Remove', key='-REMOVE-', size=(15, 1)),
                 sg.Button('🗑 Clear Queue', key='-CLEAR_QUEUE-', size=(15, 1))],
                [sg.Button('▶️ Start Download', key='-START-', size=(15, 1)),
                 sg.Button('⏹ Cancel', key='-CANCEL-', size=(15, 1)),
                 sg.Button('⚙️ Settings', key='-SETTINGS-', size=(15, 1))]
            ])]
        ]

        # Progress Section
        progress_section = [
            [sg.Frame('Progress', [
                [sg.ProgressBar(100, orientation='h', size=(50, 20),
                              key='-PROGRESS-')],
                [sg.Text('Status:', font=('Helvetica', 10)),
                 sg.Text('Waiting...', key='-STATUS-', size=(40, 1),
                        font=('Helvetica', 10))],
                [sg.Text('Speed:', font=('Helvetica', 10)),
                 sg.Text('0 KB/s', key='-SPEED-', size=(15, 1),
                        font=('Helvetica', 10)),
                 sg.Text('ETA:', font=('Helvetica', 10)),
                 sg.Text('--:--', key='-ETA-', size=(15, 1),
                        font=('Helvetica', 10))]
            ])]
        ]

        # History Section
        history_section = [
            [sg.Frame('Download History', [
                [sg.Multiline(size=(70, 5), key='-HISTORY-', disabled=True,
                            autoscroll=True, font=('Courier', 10))],
                [sg.Button('📤 Export History', key='-EXPORT_HISTORY-',
                          size=(15, 1)),
                 sg.Button('🗑 Clear History', key='-CLEAR_HISTORY-',
                          size=(15, 1))]
            ])]
        ]

        # Footer Section
        footer_section = [
            [sg.Button('📂 Open Downloads', key='-OPEN_FOLDER-'),
             sg.Button('ℹ️ About', key='-ABOUT-'),
             sg.Button('❌ Exit', key='-EXIT-')],
            [sg.Text(f'v{VERSION}', font=('Helvetica', 8), pad=((5,5),(5,5)))]
        ]

        # Combine all sections
        layout = [
            url_section,
            options_section,
            queue_section,
            progress_section,
            history_section,
            footer_section
        ]

        return sg.Window(
            'YouTube Downloader',
            layout,
            finalize=True,
            return_keyboard_events=True,
            resizable=True,
            icon=self._get_icon(),
            enable_close_attempted_event=True
        )

    def _create_tray(self) -> Optional[sg.SystemTray]:
        """Create system tray icon with error handling"""
        try:
            menu = ['', ['Show Window', 'Hide Window', '---', 'Exit']]
            return sg.SystemTray(menu, filename=self._get_icon())
        except Exception as e:
            logger.warning(f"Failed to create system tray: {e}")
            return None

    def _get_icon(self) -> Optional[bytes]:
        """Get application icon"""
        try:
            icon_path = Path(__file__).parent / 'assets' / 'icon.png'
            if icon_path.exists():
                return icon_path.read_bytes()
        except Exception:
            return None
        return None

    def run(self):
        """Main application loop with improved event handling"""
        try:
            while True:
                # Handle system tray events
                if self.tray:
                    tray_event = self.tray.read(timeout=100)
                    self._handle_tray_event(tray_event)

                # Handle window events
                event, values = self.window.read(timeout=100)

                # Handle window close
                if event in (None, '-EXIT-'):
                    if self._confirm_exit():
                        break
                    continue

                # Handle keyboard shortcuts
                event = self._handle_keyboard_shortcuts(event)

                # Handle other events
                self._handle_event(event, values)

        except Exception as e:
            logger.critical(f"Application crash: {e}", exc_info=True)
            sg.popup_error(f"Critical error: {e}nCheck logs for details.")
        finally:
            self._cleanup()

    def _handle_event(self, event: str, values: Dict[str, Any]):
        """Handle window events"""
        try:
            # Map events to handler methods
            handlers = {
                '-DETECT-': self._handle_detect,
                '-PREVIEW-': self._handle_preview,
                '-ADD-': self._handle_add,
                '-REMOVE-': self._handle_remove,
                '-START-': self._handle_start,
                '-CANCEL-': self._handle_cancel,
                '-SETTINGS-': self._handle_settings,
                '-ABOUT-': self._handle_about,
                '-OPEN_FOLDER-': self._handle_open_folder,
                '-EXPORT_HISTORY-': self._handle_export_history,
                '-CLEAR_HISTORY-': self._handle_clear_history,
                '-PASTE-': self._handle_paste,
                '-PROGRESS-': self._handle_progress,
                '-FINISHED-': self._handle_finished,
                '-DOWNLOAD_ERROR-': self._handle_error,
                '-UPDATE_QUEUE-': self._handle_queue_update,
                '-QUEUE_FINISHED-': self._handle_queue_finished
            }

            # Call appropriate handler
            if event in handlers:
                handlers[event](values)

        except Exception as e:
            logger.error(f"Error handling event {event}: {e}")
            sg.popup_error(f"Error: {e}")

    # Event handler methods...
    # (Implementation of all the _handle_* methods would go here)

    def _cleanup(self):
        """Clean up resources before exit"""
        try:
            # Save current state
            if self.config_manager.config.auto_queue:
                self.queue_manager.save_queue(
                    list(self.download_manager.download_queue.queue)
                )

            # Save configuration
            self.config_manager.save_config()

            # Cancel downloads
            self.download_manager.cancel_all_downloads()

            # Clean temporary files
            FileManager.cleanup_temp_files()

            # Close window and tray
            self.window.close()
            if self.tray:
                self.tray.close()

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Application entry point with error handling"""
    try:
        # Ensure all required directories exist
        for directory in [APP_ROOT, LOG_DIR, DOWNLOADS_ROOT, TEMP_DIR, CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize and run application
        app = YoutubeDownloaderGUI()
        app.run()

    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        sg.popup_error(
            "Failed to start application. Check logs for details.",
            title="Critical Error"
        )
    finally:
        # Ensure logs are flushed
        logging.shutdown()

if __name__ == '__main__':
    main()


# This script is a YouTube downloader application with enhanced error handling, logging, and GUI features.
# It uses yt-dlp for downloading videos and supports various formats and options.
# The application is designed to be user-friendly and includes a system tray icon for easy access.
# The script is structured to allow for easy maintenance and future enhancements.

# It includes classes for managing downloads, file operations, and system interactions.
# The application also includes a configuration manager for storing and loading settings.
# The script uses a logging system to track events and errors, and it includes a GUI with buttons and input fields for user interaction.
# The script is designed to be modular, with separate classes for different functionalities.
# It includes a main function for running the application and error handling for unexpected events.

