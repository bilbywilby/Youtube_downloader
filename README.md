# YouTube Downloader (PySimpleGUI + yt-dlp)

A modern, user-friendly desktop YouTube downloader written in Python.  
Features download queue, per-day file organization, configurable settings, history, and robust error handling.

## Features

- Download individual videos or playlists from YouTube (and most supported sites)
- Video or audio (MP3, WAV, M4A) extraction
- Date-organized downloads (`downloads/YYYY-MM-DD/`)
- Persistent download queue and history
- Customizable settings: concurrent downloads, rate limits, retries, disk space checks, and more
- System tray integration and keyboard shortcuts
- Exportable download history
- Cross-platform (Windows, macOS, Linux)
- All configuration and data stored locally in `~/.youtube_downloader/`

## Usage

1. **Install Requirements**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the Application**  
   ```
   python youtube_downloader.py
   ```

3. **Configuration**  
   Settings are available via the ⚙️ Settings button in the main window.  
   All downloads and app data are stored in `~/.youtube_downloader/` by default.

## Folder Structure

```
.yt_downloader/
├── config.json
├── history.json
├── queue.json
├── logs/
│   └── downloader.log
├── downloads/
│   └── 2025-04-16/
│       ├── VideoTitle.mp4
│       └── ...
├── temp/
└── youtube_downloader.py
```

## Requirements

- Python 3.8 or newer
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [PySimpleGUI](https://pysimplegui.readthedocs.io/)
- [psutil](https://pypi.org/project/psutil/)

## Security and Privacy

- All downloads and personal data remain on your computer.
- The application does not send telemetry or usage data.
- All paths and filenames are sanitized for user safety.

## Troubleshooting

- **ffmpeg**: For audio extraction or advanced downloads, [ffmpeg](https://ffmpeg.org/) should be installed and available in your PATH.
- **Permissions**: If you have issues creating files, check your permissions for `~/.youtube_downloader/`.
- **Updates**: yt-dlp updates frequently to counter site changes. To upgrade:  
  ```
  pip install -U yt-dlp
  ```

## License

MIT License

## Credits

- Built with [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI).