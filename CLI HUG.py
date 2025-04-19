#!/usr/bin/env python3
import os
import sys
import traceback
from pathlib import Path
import time # For potential pauses

try:
    # Import necessary functions from huggingface_hub
    from huggingface_hub import hf_hub_download, snapshot_download, model_info
    from huggingface_hub.utils import HfHubHTTPError, HFValidationError
except ImportError:
    print(
        "Error: huggingface_hub library not found.\n"
        "Please install it using: pip install huggingface_hub",
        file=sys.stderr
    )
    sys.exit(1)

try:
    # Import rich for styled output
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, IntPrompt, Confirm # For better prompting
except ImportError:
    print(
        "Error: rich library not found.\n"
        "Please install it using: pip install rich",
        file=sys.stderr
    )
    # Define fallback simple input if rich prompt is not available
    class FallbackPrompt:
        @staticmethod
        def ask(prompt, default=None, **kwargs):
            full_prompt = prompt
            if default is not None:
                full_prompt += f" [default: {default}]"
            full_prompt += ": "
            value = input(full_prompt)
            if not value and default is not None:
                return default
            return value

    class FallbackIntPrompt(FallbackPrompt):
         @staticmethod
         def ask(prompt, default=None, **kwargs):
            while True:
                val_str = FallbackPrompt.ask(prompt, default=str(default) if default is not None else None)
                try:
                    return int(val_str)
                except ValueError:
                    print("Invalid input. Please enter an integer.")

    class FallbackConfirm(FallbackPrompt):
         @staticmethod
         def ask(prompt, default=False, **kwargs):
             while True:
                val_str = FallbackPrompt.ask(prompt, default="y" if default else "n")
                if val_str.lower() in ['y', 'yes']: return True
                if val_str.lower() in ['n', 'no']: return False
                print("Please enter 'y' or 'n'.")

    class FallbackConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("-" * 20)
        def status(self, *args, **kwargs): return self # Dummy status context
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def update(self, *args, **kwargs): print(args[0] if args else "Working...")

    Console = FallbackConsole # type: ignore
    Prompt = FallbackPrompt # type: ignore
    IntPrompt = FallbackIntPrompt # type: ignore
    Confirm = FallbackConfirm # type: ignore


# Initialize rich console
console = Console()

# --- Default Settings ---
DEFAULT_SAVE_DIR = Path.home() / "hf_models"
DEFAULT_WORKERS = 3
# Define default models/files for prompts
DEFAULT_SINGLE_FILE_REPO = "google-bert/bert-base-uncased"
DEFAULT_SINGLE_FILENAME = "config.json"
DEFAULT_MODEL_REPO = "distilbert-base-uncased"


# --- Helper Functions ---

def display_main_menu():
    """Prints the main menu options."""
    console.print(Panel(
        Text("1. Download Single File\n"
             "2. Download Entire Model\n"
             "3. Exit", justify="left"),
        title="[bold cyan]Hugging Face Downloader Menu[/bold cyan]",
        border_style="blue",
        padding=(1, 2)
    ))

def handle_download_error(e, repo_id, filename=None):
    """Handles common download errors and prints styled messages."""
    if isinstance(e, HfHubHTTPError):
        if "404" in str(e) or "not found" in str(e).lower() or "repository not found" in str(e).lower():
            target = f"File '{filename}' in repo" if filename else "Repository"
            console.print(f"[bold red]Error:[/bold red] {target} '{repo_id}' not found (404). Please check the names.")
        else:
            console.print(f"[bold red]Error:[/bold red] Network or server error downloading from '{repo_id}'.")
            console.print(f"[dim]{e}[/dim]")
    elif isinstance(e, HFValidationError):
         console.print(f"[bold red]Error:[/bold red] Invalid repository or file name: '{repo_id}'{f'/{filename}' if filename else ''}.")
         console.print(f"[dim]{e}[/dim]")
    elif isinstance(e, FileNotFoundError):
         console.print(f"[bold red]Error:[/bold red] Could not create local directory. Check permissions.")
         console.print(f"[dim]{e}[/dim]")
    else:
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        # Print traceback for unexpected errors only if they are not HfHubHTTPError or HFValidationError
        # as those are handled more gracefully above.
        if not isinstance(e, (HfHubHTTPError, HFValidationError)):
             console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
             console.print(f"[dim]{e}[/dim]") # Print the error string for handled types

def ensure_directory(path: Path):
    """Ensures a directory exists, creating it if necessary. Returns True on success."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Could not create directory '{path}'. Check permissions.")
        console.print(f"[dim]{e}[/dim]")
        return False

def run_single_download():
    """Prompts for single file info and initiates download."""
    # Applying user's color preference for the rule
    console.rule("[#ADFF2F]Download Single File[/#ADFF2F]")
    repo_id = Prompt.ask("[cyan]Enter Repository ID[/cyan]", default=DEFAULT_SINGLE_FILE_REPO)
    if not repo_id: console.print("[yellow]Repo ID cannot be empty. Aborting.[/yellow]"); return

    filename = Prompt.ask("[cyan]Enter Filename[/cyan]", default=DEFAULT_SINGLE_FILENAME)
    if not filename: console.print("[yellow]Filename cannot be empty. Aborting.[/yellow]"); return

    local_dir_str = Prompt.ask(f"[cyan]Enter Save Directory[/cyan]", default=str(DEFAULT_SAVE_DIR))
    local_dir = Path(local_dir_str)

    if not ensure_directory(local_dir): return

    console.print(f"\n[magenta]Starting download...[/magenta]")
    # Applying user's color preference for info lines
    console.print(f"  Repo ID: [#ADFF2F]{repo_id}[/#ADFF2F]")
    console.print(f"  Filename: [#ADFF2F]{filename}[/#ADFF2F]")
    console.print(f"  Save Dir: [#ADFF2F]{local_dir}[/#ADFF2F]")
    console.rule()

    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            cache_dir=local_dir / ".cache"
        )
        console.rule()
        # Applying user's color preference for success message
        console.print(f"[#ADFF2F]Success![/#ADFF2F] File downloaded to:")
        # Applying user's color preference for path and FIXING the closing tag
        console.print(f"  [bold #F59E0B]{file_path}[/bold #F59E0B]") # <-- FIX HERE

    except Exception as e:
        console.rule()
        handle_download_error(e, repo_id, filename)

def run_model_download():
    """Prompts for model info and initiates download."""
    console.rule("[bold blue]Download Entire Model[/bold blue]")
    repo_id = Prompt.ask("[cyan]Enter Repository ID[/cyan]", default=DEFAULT_MODEL_REPO)
    if not repo_id: console.print("[yellow]Repo ID cannot be empty. Aborting.[/yellow]"); return

    local_dir_base_str = Prompt.ask(f"[cyan]Enter Base Save Directory[/cyan]", default=str(DEFAULT_SAVE_DIR))
    local_dir_base = Path(local_dir_base_str)

    workers = IntPrompt.ask(f"[cyan]Enter Number of Workers[/cyan]", default=DEFAULT_WORKERS, choices=[str(w) for w in [1, 2, 3, 4, 6, 8]])

    if not ensure_directory(local_dir_base): return

    # --- Get File Count ---
    num_files_str = ""
    try:
        with console.status(f"[cyan]Fetching model info for {repo_id}...[/]"):
             info = model_info(repo_id=repo_id)
        num_files = len(info.siblings)
        num_files_str = f" ({num_files} files expected)"
        console.print(f"  Model Info: [green]Found {num_files} files.[/green]")
    except Exception as info_err:
         console.print(f"  Model Info: [yellow]Warning: Could not get file count: {info_err}[/yellow]")
    # ---

    console.print(f"\n[magenta]Starting download...[/magenta]")
    console.print(f"  Repo ID: [bold]{repo_id}[/bold]") # Keeping original bold for contrast
    console.print(f"  Save Dir Base: [bright_green]{local_dir_base}[/bright_green]")
    # FIXING closing tag typo here
    console.print(f"  Workers: [bright_green]{workers}[/bright_green]") # <-- FIX HERE
    console.rule(f"Starting Download{num_files_str}")

    model_target_dir = local_dir_base / repo_id

    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_target_dir,
            max_workers=workers
        )
        console.rule()
        console.print(f"[green]Success![/green] Model downloaded to directory:")
        # Keeping original bold cyan for contrast
        console.print(f"  [bold cyan]{model_path}[/bold cyan]")

    except Exception as e:
        console.rule()
        handle_download_error(e, repo_id)


# --- Main Application Loop ---
def run_app():
    """Runs the main interactive menu loop."""
    while True:
        display_main_menu()
        choice = Prompt.ask("[bold #F59E0B]Choose an option[/bold #F59E0B]", choices=["1", "2", "3"])

        if choice == '1':
            run_single_download()
        elif choice == '2':
            run_model_download()
        elif choice == '3':
            console.print("[bold blue]Exiting program.[/bold blue]")
            break
        else:
            console.print("[red]Invalid choice, please try again.[/red]")

        console.print("\nPress Enter to continue...")
        input()
        console.print("\n" * 2)


if __name__ == "__main__":
    try:
        run_app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user. Exiting.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print("[bold red]An unexpected critical error occurred:[/bold red]")
        # Print simpler error for known handled types, full traceback otherwise
        if not isinstance(e, (HfHubHTTPError, HFValidationError, FileNotFoundError)):
             console.print_exception(show_locals=False)
        else:
             # Error should have been handled already by handle_download_error
             # This is a fallback print
             console.print(f"[dim]{e}[/dim]")
        sys.exit(1)
