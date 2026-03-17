import sys
import time
from unittest.mock import patch
import io
import importlib.util

# Simulate user inputs, now answering "No" to the new "Fetch file count?" prompt
inputs = [
    "2", # Option 2: Download Entire Model
    "distilbert-base-uncased",  # Default Repo ID
    "/tmp",  # Default Save Dir
    "3",  # Default Workers
    "n",  # Do not fetch file count
    "3", # Option 3: Exit
]

def mock_input(prompt=None):
    if inputs:
        return inputs.pop(0)
    return ""

def mock_snapshot_download(*args, **kwargs):
    return "dummy_path"

with patch('builtins.input', mock_input):
    with patch('huggingface_hub.snapshot_download', mock_snapshot_download):
        spec = importlib.util.spec_from_file_location("cli_hug", "CLI HUG.py")
        cli_hug = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_hug)

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        start_time = time.time()
        try:
            with patch('rich.prompt.Prompt.ask', side_effect=inputs):
                with patch('rich.prompt.IntPrompt.ask', return_value=3):
                    # We also need to patch Confirm.ask specifically since we use it now
                    # We'll patch it to return False
                    with patch('rich.prompt.Confirm.ask', return_value=False):
                         cli_hug.run_app()
        except SystemExit:
            pass
        except StopIteration:
            pass

        end_time = time.time()
        sys.stdout = original_stdout

        print(f"Optimized Time taken (No to info fetch): {end_time - start_time:.4f} seconds")
