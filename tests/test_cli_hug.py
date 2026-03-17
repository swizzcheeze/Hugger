import sys
import os
from unittest.mock import MagicMock, patch
import pytest
import importlib.util

# Mocking the dependencies before importing the module under test
class HfHubHTTPError(Exception):
    pass

class HFValidationError(Exception):
    pass

mock_hf_hub = MagicMock()
mock_hf_hub_utils = MagicMock()
mock_hf_hub_utils.HfHubHTTPError = HfHubHTTPError
mock_hf_hub_utils.HFValidationError = HFValidationError

sys.modules["huggingface_hub"] = mock_hf_hub
sys.modules["huggingface_hub.utils"] = mock_hf_hub_utils

# Mock rich to ensure it follows the rich path if it were installed
# or at least doesn't break.
sys.modules["rich"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["rich.panel"] = MagicMock()
sys.modules["rich.text"] = MagicMock()
sys.modules["rich.prompt"] = MagicMock()

# Now import the module with space in filename
spec = importlib.util.spec_from_file_location("cli_hug", "CLI HUG.py")
cli_hug = importlib.util.module_from_spec(spec)
sys.modules["cli_hug"] = cli_hug
spec.loader.exec_module(cli_hug)

@pytest.fixture
def mock_console():
    with patch("cli_hug.console") as mock:
        yield mock

def test_handle_download_error_404_repository(mock_console):
    error = HfHubHTTPError("404 Client Error: Not Found for url")
    cli_hug.handle_download_error(error, "some/repo")

    mock_console.print.assert_called_once_with("[bold red]Error:[/bold red] Repository 'some/repo' not found (404). Please check the names.")

def test_handle_download_error_404_file(mock_console):
    error = HfHubHTTPError("404 Client Error: Not Found for url")
    cli_hug.handle_download_error(error, "some/repo", filename="config.json")

    mock_console.print.assert_called_once_with("[bold red]Error:[/bold red] File 'config.json' in repo 'some/repo' not found (404). Please check the names.")

def test_handle_download_error_other_http_error(mock_console):
    error = HfHubHTTPError("500 Server Error")
    cli_hug.handle_download_error(error, "some/repo")

    assert mock_console.print.call_count == 2
    mock_console.print.assert_any_call("[bold red]Error:[/bold red] Network or server error downloading from 'some/repo'.")
    mock_console.print.assert_any_call("[dim]500 Server Error[/dim]")

def test_handle_download_error_validation_error(mock_console):
    error = HFValidationError("Invalid repo name")
    cli_hug.handle_download_error(error, "invalid/repo")

    assert mock_console.print.call_count == 2
    mock_console.print.assert_any_call("[bold red]Error:[/bold red] Invalid repository or file name: 'invalid/repo'.")
    mock_console.print.assert_any_call("[dim]Invalid repo name[/dim]")

def test_handle_download_error_file_not_found(mock_console):
    error = FileNotFoundError("No such file or directory")
    cli_hug.handle_download_error(error, "some/repo")

    assert mock_console.print.call_count == 2
    mock_console.print.assert_any_call("[bold red]Error:[/bold red] Could not create local directory. Check permissions.")
    mock_console.print.assert_any_call("[dim]No such file or directory[/dim]")

def test_handle_download_error_unexpected_exception(mock_console):
    try:
        raise ValueError("Something went wrong")
    except ValueError as error:
        cli_hug.handle_download_error(error, "some/repo")

    assert mock_console.print.call_count == 2
    mock_console.print.assert_any_call("[bold red]An unexpected error occurred:[/bold red]")
    # The second call should contain the traceback
    args, _ = mock_console.print.call_args_list[1]
    assert "[dim]" in args[0]
    assert "Traceback" in args[0]
    assert "ValueError: Something went wrong" in args[0]
