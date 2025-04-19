import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, scrolledtext, Menu
import tkinter.font as tkFont
import os
import threading
import queue
import sys
import traceback
import re
import subprocess

# --- Color Scheme (Dark theme inspired by Hugging Face) ---
BG_COLOR = '#2D3748'  # Dark Gray-Blue (Window, Frames)
FG_COLOR = '#E2E8F0'  # Light Gray (Default Text)
BG_LABELFRAME = '#4A5568' # Slightly Lighter Gray for LabelFrame title background
ENTRY_BG = '#4A5568'  # Medium Gray (Entry/Combobox Background)
ENTRY_FG = '#F7FAFC'  # Very Light Gray (Entry/Combobox Text)
BUTTON_BG = '#4A5568'  # Medium Gray (Button Background)
BUTTON_FG = FG_COLOR   # Light Gray (Button Text)
BUTTON_ACTIVE_BG = '#718096' # Lighter Gray (Button Hover/Active)
BUTTON_CANCEL_BG = '#E53E3E' # Red for Cancel button background
BUTTON_CANCEL_FG = '#FFFFFF' # White for Cancel button text
BUTTON_CANCEL_ACTIVE_BG = '#FC8181' # Lighter Red for Cancel hover/active
PROGRESS_TROUGH = '#4A5568' # Trough color for progress bar
PROGRESS_BAR = '#63B3ED'   # Blue accent for progress bar
TEXT_BG = '#1A202C'  # Very Dark Gray-Blue (Log Area Background)
LOG_TEXT_FG = '#F59E0B' # Amber/Gold color for Log Text
SELECT_BG = '#4299E1' # Blue for selected text background
SELECT_FG = '#FFFFFF' # White for selected text foreground


# Simple import with minimal dependencies
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    print("Successfully imported core huggingface_hub functions")
except ImportError as e:
    print(f"Import Error: {e}")
    root_check = tk.Tk(); root_check.withdraw()
    messagebox.showerror("Error", f"Could not import from huggingface_hub: {e}\n\nPlease install with:\npip install huggingface_hub")
    root_check.destroy(); sys.exit(1)
except Exception as e:
    print(f"Unexpected Import Error: {e}")
    root_check = tk.Tk(); root_check.withdraw()
    messagebox.showerror("Error", f"An unexpected error occurred during import:\n{e}")
    root_check.destroy(); sys.exit(1)

# --- Threaded Download Functions ---
# Modified slightly to check cancellation flag (though cannot interrupt mid-download)
def download_single_file_threaded(model_id, filename, local_dir, status_queue, cancel_event):
    """Downloads a single file in a thread and reports status via queue."""
    status_queue.put(("PROGRESS_START", f"Downloading {filename}..."))
    file_path = None # Initialize file_path
    try:
        if cancel_event.is_set(): raise InterruptedError("Download cancelled before start") # Check before starting
        status_queue.put(f"Starting download: {filename} from {model_id}...")
        os.makedirs(local_dir, exist_ok=True)

        # NOTE: hf_hub_download itself cannot be easily interrupted by the flag here.
        # The cancellation primarily works by ignoring the result later.
        file_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=local_dir
            # Removed deprecated local_dir_use_symlinks=False
        )
        # Check cancellation *after* download completes but *before* sending success
        if cancel_event.is_set(): raise InterruptedError("Download cancelled during operation")

        status_queue.put(f"SUCCESS: Downloaded {filename} from {model_id}\nSaved to: {file_path}")

    except InterruptedError:
         status_queue.put(f"INFO: Download task for {filename} acknowledged cancellation.")
    except Exception as e:
        # Avoid sending error if cancellation happened during the download
        if not cancel_event.is_set():
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower() or "repository not found" in error_msg.lower(): status_queue.put(f"ERROR: File or Repository not found.\nModel: {model_id}\nFile: {filename}\nDetails: {e}")
            else: status_queue.put(f"ERROR: Download failed.\nDetails: {e}\n{traceback.format_exc()}")
        else:
             status_queue.put(f"INFO: Download task for {filename} failed after cancellation request.")
    finally:
        status_queue.put("PROGRESS_END")
        status_queue.put("DONE_SINGLE") # Always signal completion

def download_entire_model_threaded(model_id, local_dir_base, num_workers, status_queue, cancel_event):
    """Downloads an entire model in a thread and reports status via queue."""
    status_queue.put(("PROGRESS_START", f"Downloading {model_id} (workers={num_workers})..."))
    model_path = None # Initialize
    try:
        if cancel_event.is_set(): raise InterruptedError("Download cancelled before start")
        status_queue.put(f"Starting download of entire model: {model_id} using {num_workers} workers...")
        model_target_dir = os.path.join(local_dir_base, model_id); os.makedirs(model_target_dir, exist_ok=True)

        # NOTE: snapshot_download itself cannot be easily interrupted by the flag here.
        model_path = snapshot_download(
            repo_id=model_id,
            local_dir=model_target_dir,
            # Removed deprecated local_dir_use_symlinks=False
            max_workers=num_workers
            # Ideally, snapshot_download would accept a cancellation token/event
        )
        if cancel_event.is_set(): raise InterruptedError("Download cancelled during operation")

        status_queue.put(f"SUCCESS: Downloaded entire model {model_id}\nSaved to: {model_path}")

    except InterruptedError:
         status_queue.put(f"INFO: Download task for {model_id} acknowledged cancellation.")
    except Exception as e:
        if not cancel_event.is_set():
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower() or "repository not found" in error_msg.lower(): status_queue.put(f"ERROR: Model or Repository not found.\nModel: {model_id}\nDetails: {e}")
            else: status_queue.put(f"ERROR: Download failed.\nDetails: {e}\n{traceback.format_exc()}")
        else:
            status_queue.put(f"INFO: Download task for {model_id} failed after cancellation request.")
    finally:
         status_queue.put("PROGRESS_END")
         status_queue.put("DONE_MODEL") # Always signal completion


# --- GUI Application Class ---
class HuggingFaceDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hugging Face Downloader")
        self.root.geometry("700x950")
        self.root.config(bg=BG_COLOR)

        # --- Style Configuration ---
        self.style = ttk.Style(self.root)
        try: self.style.theme_use('clam')
        except tk.TclError: print("Clam theme not available, using default.")

        # --- Fonts ---
        try:
            default_font = tkFont.nametofont("TkDefaultFont"); self.default_font_family = default_font.actual("family"); self.default_font_size = default_font.actual("size")
            self.bold_font = tkFont.Font(family=self.default_font_family, size=self.default_font_size, weight="bold")
        except tk.TclError: self.default_font_family = 'Helvetica'; self.default_font_size = 10; self.bold_font = (self.default_font_family, self.default_font_size, 'bold')

        # --- Apply Styles ---
        self.style.configure('.', background=BG_COLOR, foreground=FG_COLOR, font=(self.default_font_family, self.default_font_size))
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabelFrame', background=BG_COLOR, borderwidth=1, relief=tk.SOLID)
        self.style.configure('TLabelFrame.Label', background=BG_LABELFRAME, foreground=FG_COLOR, font=self.bold_font, padding=(5, 2))
        self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR)
        self.style.configure('TButton', background=BUTTON_BG, foreground=BUTTON_FG, padding=5, font=(self.default_font_family, self.default_font_size))
        self.style.map('TButton', background=[('active', BUTTON_ACTIVE_BG), ('pressed', BUTTON_ACTIVE_BG), ('disabled', BG_LABELFRAME)])
        self.style.configure('Cancel.TButton', background=BUTTON_CANCEL_BG, foreground=BUTTON_CANCEL_FG, padding=5, font=(self.default_font_family, self.default_font_size))
        self.style.map('Cancel.TButton', background=[('active', BUTTON_CANCEL_ACTIVE_BG), ('pressed', BUTTON_CANCEL_ACTIVE_BG), ('disabled', BG_LABELFRAME)])
        self.style.map('TEntry', fieldbackground=[('!disabled', ENTRY_BG)], foreground=[('!disabled', ENTRY_FG)], insertcolor=FG_COLOR)
        self.style.map('TCombobox', fieldbackground=[('!disabled', ENTRY_BG)], foreground=[('!disabled', ENTRY_FG)], selectbackground=ENTRY_BG, selectforeground=FG_COLOR, insertcolor=FG_COLOR, arrowcolor=FG_COLOR)
        self.root.option_add('*TCombobox*Listbox.background', ENTRY_BG); self.root.option_add('*TCombobox*Listbox.foreground', FG_COLOR)
        self.root.option_add('*TCombobox*Listbox.selectBackground', SELECT_BG); self.root.option_add('*TCombobox*Listbox.selectForeground', SELECT_FG)
        self.style.configure('Horizontal.TProgressbar', troughcolor=PROGRESS_TROUGH, background=PROGRESS_BAR, thickness=15)
        self.style.configure('Vertical.TScrollbar', background=BUTTON_BG, troughcolor=BG_COLOR, bordercolor=BG_COLOR, arrowcolor=FG_COLOR)
        self.style.map('Vertical.TScrollbar', background=[('active', BUTTON_ACTIVE_BG)])

        # --- Shared Variables ---
        self.default_save_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "hf_models"))
        self.status_queue = queue.Queue()
        self.speed_selection = tk.StringVar()
        self.sf_model_id_var = tk.StringVar()
        self.em_model_id_var = tk.StringVar()
        self.cancel_requested = threading.Event()
        self.download_active = False

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10", style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # --- Status Bar Frame ---
        status_bar_frame = ttk.Frame(self.root, padding=(5, 2), style='TFrame')
        status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar_label = ttk.Label(status_bar_frame, text="Ready", anchor=tk.W, style='TLabel', width=40)
        self.status_bar_label.pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(status_bar_frame, style='Horizontal.TProgressbar', mode='determinate', length=150)
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.cancel_button = ttk.Button(status_bar_frame, text="Cancel", command=self.cancel_download, style='Cancel.TButton', state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)

        # --- Shared Save Directory Selection ---
        dir_frame = ttk.LabelFrame(main_frame, text="Save Location", padding="10")
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(dir_frame, text="Base Directory:", font=self.bold_font, style='TLabel').grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.default_save_dir, width=50, style='TEntry')
        self.dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ttk.Button(dir_frame, text="Browse...", command=self.browse_directory, style='TButton')
        self.browse_button.grid(row=0, column=2, padx=(5, 2), pady=5)
        self.open_dir_button = ttk.Button(dir_frame, text="Open...", command=self.open_download_directory, style='TButton')
        self.open_dir_button.grid(row=0, column=3, padx=(2, 5), pady=5)
        dir_frame.columnconfigure(1, weight=1)

        # --- Single File Download Section ---
        single_frame = ttk.LabelFrame(main_frame, text="Download Single File", padding="10")
        single_frame.pack(fill=tk.X, pady=5)
        ttk.Label(single_frame, text="Model ID:", font=self.bold_font, style='TLabel').grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.sf_model_id_entry = ttk.Entry(single_frame, width=50, style='TEntry', textvariable=self.sf_model_id_var)
        self.sf_model_id_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.sf_model_id_var.set("bert-base-uncased")
        ttk.Label(single_frame, text="Filename:", font=self.bold_font, style='TLabel').grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.sf_filename_entry = ttk.Entry(single_frame, width=50, style='TEntry')
        self.sf_filename_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.sf_filename_entry.insert(0, "config.json")
        self.sf_download_button = ttk.Button(single_frame, text="Download File", command=self.start_single_file_download, style='TButton')
        self.sf_download_button.grid(row=2, column=0, columnspan=3, pady=10)
        single_frame.columnconfigure(1, weight=1)

        # --- Download Speed Selection (Dropdown) ---
        speed_frame = ttk.LabelFrame(main_frame, text="Model Download Speed", padding="10")
        speed_frame.pack(fill=tk.X, pady=(10, 10))
        ttk.Label(speed_frame, text="Workers:", font=self.bold_font, style='TLabel').pack(side=tk.LEFT, padx=(0, 5))
        speed_options = ["Normal (1 worker)", "Fast (3 workers)", "Ultra (6 workers)"]
        self.speed_dropdown = ttk.Combobox(speed_frame, textvariable=self.speed_selection, values=speed_options, state="readonly", width=20, style='TCombobox')
        self.speed_dropdown.pack(side=tk.LEFT)
        self.speed_dropdown.set("Fast (3 workers)")

        # --- Entire Model Download Section ---
        model_frame = ttk.LabelFrame(main_frame, text="Download Entire Model", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Model ID:", font=self.bold_font, style='TLabel').grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.em_model_id_entry = ttk.Entry(model_frame, width=50, style='TEntry', textvariable=self.em_model_id_var)
        self.em_model_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.em_model_id_var.set(self.sf_model_id_var.get())
        self.em_download_button = ttk.Button(model_frame, text="Download Model", command=self.start_entire_model_download, style='TButton')
        self.em_download_button.grid(row=1, column=0, columnspan=2, pady=10)
        model_frame.columnconfigure(1, weight=1)

        # --- Status Log Area ---
        status_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, bg=TEXT_BG, fg=LOG_TEXT_FG, selectbackground=SELECT_BG, selectforeground=SELECT_FG, insertbackground=FG_COLOR)
        self.status_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        log_buttons_frame = ttk.Frame(status_frame, style='TFrame')
        log_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        self.clear_button = ttk.Button(log_buttons_frame, text="Clear Log", command=self.clear_status_log, style='TButton')
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.save_log_button = ttk.Button(log_buttons_frame, text="Save Log...", command=self.save_status_log, style='TButton')
        self.save_log_button.pack(side=tk.LEFT, padx=5)

        # --- Right-click Menu for Status Log ---
        self.log_context_menu = Menu(self.status_text, tearoff=0, bg=BUTTON_BG, fg=BUTTON_FG, activebackground=BUTTON_ACTIVE_BG, activeforeground=BUTTON_FG)
        self.log_context_menu.add_command(label="Copy", command=self.copy_log_text)
        self.status_text.bind("<Button-3>", self.show_log_context_menu)
        self.status_text.bind("<Button-2>", self.show_log_context_menu)

        # --- Initial Setup ---
        self.sf_model_id_var.trace_add('write', self.sync_model_ids)
        self.check_queue()
        self.update_status_bar("Ready")

    # --- Methods for GUI Actions ---
    # (Methods remain the same)
    def sync_model_ids(self, *args):
        try: self.em_model_id_var.set(self.sf_model_id_var.get())
        except Exception as e: print(f"Error syncing model IDs: {e}")

    def cancel_download(self):
        if self.download_active:
            self.log_status(">>> Cancellation Requested <<<")
            self.update_status_bar("Cancellation requested...")
            self.cancel_requested.set(); self.cancel_button.config(state=tk.DISABLED)

    def open_download_directory(self):
        dir_path = self.default_save_dir.get().strip()
        if not dir_path: messagebox.showwarning("Open Directory", "No directory path specified."); self.update_status_bar("No directory specified to open."); return
        if not os.path.isdir(dir_path): messagebox.showwarning("Open Directory", f"Directory does not exist:\n{dir_path}"); self.update_status_bar("Selected directory does not exist."); return
        try:
            self.update_status_bar(f"Opening directory: {dir_path}")
            if sys.platform == "win32": os.startfile(os.path.realpath(dir_path))
            elif sys.platform == "darwin": subprocess.call(["open", dir_path])
            else: subprocess.call(["xdg-open", dir_path])
            self.update_status_bar(f"Opened directory.")
        except FileNotFoundError: messagebox.showerror("Open Directory Error", f"Could not find command to open directory."); self.update_status_bar("Error opening directory: command not found.")
        except Exception as e: messagebox.showerror("Open Directory Error", f"Failed to open directory.\nError: {e}"); self.update_status_bar("Error opening directory."); print(f"Error opening directory: {e}\n{traceback.format_exc()}")

    def clear_status_log(self):
        self.status_text.config(state=tk.NORMAL); self.status_text.delete(1.0, tk.END); self.status_text.config(state=tk.DISABLED)

    def save_status_log(self):
        log_content = self.status_text.get(1.0, tk.END).strip()
        if not log_content: messagebox.showinfo("Save Log", "Log is empty, nothing to save."); return
        filepath = filedialog.asksaveasfilename(title="Save Status Log", defaultextension=".log", filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")], initialdir=os.getcwd())
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: f.write(log_content)
            self.update_status_bar(f"Log saved to {os.path.basename(filepath)}"); messagebox.showinfo("Save Log", f"Log successfully saved to:\n{filepath}")
        except Exception as e: messagebox.showerror("Save Log Error", f"Failed to save log file.\nError: {e}"); self.update_status_bar("Error saving log.")

    def show_log_context_menu(self, event):
        try: self.log_context_menu.tk_popup(event.x_root, event.y_root)
        finally: self.log_context_menu.grab_release()

    def copy_log_text(self):
        try: selected_text = self.status_text.get(tk.SEL_FIRST, tk.SEL_LAST); text_to_copy = selected_text
        except tk.TclError: text_to_copy = self.status_text.get(1.0, tk.END).strip()
        if text_to_copy: self.root.clipboard_clear(); self.root.clipboard_append(text_to_copy); self.update_status_bar("Log content copied to clipboard.")
        else: self.update_status_bar("Nothing to copy from log.")

    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.default_save_dir.get())
        if directory: self.default_save_dir.set(directory)

    def update_status_bar(self, message):
        self.status_bar_label.config(text=message); self.root.update_idletasks()

    def start_progress(self, message="Processing..."):
        self.update_status_bar(message); self.progress_bar.config(mode='indeterminate'); self.progress_bar.start(10)

    def stop_progress(self, final_message="Ready"):
        self.progress_bar.stop(); self.progress_bar.config(mode='determinate', value=0); self.update_status_bar(final_message)

    def log_status(self, message):
        if not self.download_active or not self.cancel_requested.is_set() or "cancel" in message.lower():
             self.status_text.config(state=tk.NORMAL); self.status_text.insert(tk.END, message + "\n\n"); self.status_text.see(tk.END); self.status_text.config(state=tk.DISABLED)

    def _start_download_ui_updates(self):
        self.download_active = True; self.cancel_requested.clear()
        self.sf_download_button.config(state=tk.DISABLED); self.em_download_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)

    def _end_download_ui_updates(self, status_message="Ready"):
        self.download_active = False
        self.sf_download_button.config(state=tk.NORMAL); self.em_download_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.stop_progress(status_message); self.cancel_requested.clear()

    def start_single_file_download(self):
        if self.download_active: return
        model_id = self.sf_model_id_var.get().strip(); filename = self.sf_filename_entry.get().strip(); local_dir = self.default_save_dir.get().strip()
        if not model_id or not filename or not local_dir: messagebox.showwarning("Input Error", "Please enter Model ID, Filename, and select a Save Directory."); return
        self._start_download_ui_updates(); self.log_status(f"Queueing single file download: {filename} from {model_id}")
        threading.Thread(target=download_single_file_threaded, args=(model_id, filename, local_dir, self.status_queue, self.cancel_requested), daemon=True).start()

    def start_entire_model_download(self):
        if self.download_active: return
        model_id = self.em_model_id_var.get().strip(); local_dir_base = self.default_save_dir.get().strip(); selection_str = self.speed_selection.get()
        workers = 3
        try:
            match = re.search(r'\d+', selection_str)
            if match: workers = int(match.group(0))
            else: self.log_status(f"Warning: Could not parse worker count from '{selection_str}'. Defaulting to {workers}.")
        except Exception as parse_err: self.log_status(f"Warning: Error parsing worker count '{selection_str}'. Defaulting to {workers}. Error: {parse_err}")
        if not model_id or not local_dir_base: messagebox.showwarning("Input Error", "Please enter Model ID and select a Base Save Directory."); return
        self._start_download_ui_updates(); self.log_status(f"Queueing entire model download: {model_id} ({workers} workers)")
        threading.Thread(target=download_entire_model_threaded, args=(model_id, local_dir_base, workers, self.status_queue, self.cancel_requested), daemon=True).start()

    def check_queue(self):
        try:
            while True:
                message_item = self.status_queue.get_nowait()
                if isinstance(message_item, tuple) and message_item[0] == "PROGRESS_START":
                    if not self.cancel_requested.is_set(): self.start_progress(message_item[1])
                    continue
                if message_item == "DONE_SINGLE" or message_item == "DONE_MODEL":
                    final_status = "Ready";
                    if self.cancel_requested.is_set(): self.log_status("--- Download Cancelled by User. ---"); final_status = "Cancelled"
                    self._end_download_ui_updates(final_status); continue
                if message_item == "PROGRESS_END":
                    if not self.download_active: self.stop_progress()
                    continue
                if not self.cancel_requested.is_set():
                    self.log_status(message_item)
                    if isinstance(message_item, str):
                        first_line = message_item.split('\n', 1)[0]
                        if "ERROR:" in first_line or "SUCCESS:" in first_line: self.update_status_bar(first_line)
        except queue.Empty: pass
        finally: self.root.after(100, self.check_queue)

# --- Run the Application ---
if __name__ == "__main__":
    try:
        print("Starting GUI...")
        root = tk.Tk()
        import re; import subprocess; import tkinter.font as tkFont
        app = HuggingFaceDownloaderApp(root)
        root.mainloop()
        print("GUI closed.")
    except Exception as e:
        print(f"Fatal Application Error: {e}\n{traceback.format_exc()}")
        try: root_err = tk.Tk(); root_err.withdraw(); messagebox.showerror("Application Error", f"An unexpected error occurred: {e}"); root_err.destroy()
        except Exception as e2: print(f"Could not show final error dialog: {e2}")
        sys.exit(1)
