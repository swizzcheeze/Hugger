# Hugging Face Hub Downloader (GUI)

A simple graphical desktop application written in Python With AI, Namely Gemini and a Touch of Claude this uses Tkinter to download single files or entire model repositories from the Hugging Face Hub. (Most this was wrtten with AI as-well)

## Features

- **Graphical User Interface:** Easy-to-use interface built with Tkinter/ttk.
- **Single File Download:** Download specific files (e.g., `config.json`).
- **Entire Model Download:** Download complete model snapshots.
- **Directory Selection:** Browse for and select the base save directory.
- **Open Directory:** Button to quickly open the selected save directory in your system's file explorer.
- **Worker Selection:** Choose download concurrency (1, 3, or 6 workers) for entire model downloads via a dropdown menu.
- **Model ID Sync:** Automatically copies the Model ID from the single file section to the entire model section as you type.
- **Status Log:** Displays detailed status messages, errors, and success confirmations.
- **Clear Log:** Button to clear the log area.
- **Save Log:** Button to save the log content to a text file.
- **Copy:** Right-click context menu to copy log content.
- **Status Bar:** Shows current activity status and an indeterminate progress bar during downloads.
- **Cancel Download:** Button to request cancellation of an ongoing download (best-effort, stops the app from processing results).
- **Theming:** Attempts a dark theme inspired by Hugging Face (appearance may vary by OS).
- **Caching:** Leverages the `huggingface_hub` library's caching mechanism to avoid re-downloading existing valid files.

## Requirements

- Python 3.x (Tkinter is usually included with standard Python installations on Windows and macOS; may need separate installation on some Linux distros, e.g., `sudo apt-get install python3-tk`).
- Required library: `huggingface_hub`

## Installation

1. Clone this repository or download the Python script (e.g., `hf_gui_downloader.py`).
2. Navigate to the script's directory in your terminal or command prompt.
3. (Highly Recommended) Create and activate a Python virtual environment:

   ```bash
   # Create the environment (e.g., named .venv)
   python -m venv .venv

   # Activate it:
   # Windows (Command Prompt/PowerShell)
   .\.venv\Scripts\activate
   # macOS/Linux (bash/zsh)
   source .venv/bin/activate
   ```
4. Install the required library:

   ```bash
   pip install huggingface_hub
   ```

## Usage

1. Make sure your virtual environment (if created) is activated.
2. Run the script from your terminal:

   ```bash
   python hf_gui_downloader.py
   ```

   (Replace `hf_gui_downloader.py` with the actual filename if you renamed it). Alternatively, on some systems, you might be able to run it by double-clicking the `.py` file if your Python installation is configured correctly.
3. The application window will appear.
4. **Save Location:** Use the "Browse..." button to select the main folder where models/files will be saved (defaults to `hf_models` in your home directory). Use the "Open..." button to view the selected folder in your file explorer.
5. **Download Single File:** Enter the Model ID and Filename, then click the "Download File" button.
6. **Model Download Speed:** Select the desired number of workers from the dropdown menu. This setting only affects the "Download Entire Model" action.
7. **Download Entire Model:** Enter the Model ID (note: it might be auto-filled as you type in the single file section). Click the "Download Model" button. A sub-folder named after the Model ID will be created in your selected save location containing the model files.
8. **Status Log:** Monitor detailed progress messages and results in the main text area. Use the "Clear Log" or "Save Log..." buttons below it, or right-click within the log area to copy text.
9. **Status Bar:** Observe the current activity message at the bottom-left and the moving progress bar during downloads. Use the "Cancel" button at the bottom-right to stop the application from waiting for the current download.
10. Close the window when finished.

## Limitations

- **Progress Bar:** The progress bar is indeterminate (shows activity but not %) because the underlying library functions do not provide the real-time byte-level progress data required for accurate percentage display within the GUI.
- **Cancellation:** The "Cancel" button requests cancellation. It stops the application from processing the download's final result but may not immediately halt the background network activity managed by the library. Downloaded files are not deleted upon cancellation.
- **Styling:** The dark theme appearance might vary across different operating systems (Windows, macOS, Linux) and desktop environments due to differences in how Tkinter/ttk interacts with native themes.

## License
