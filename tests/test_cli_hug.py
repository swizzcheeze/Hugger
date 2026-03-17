import pytest
import importlib.util
from pathlib import Path

# Load CLI HUG.py module dynamically to bypass the space in the filename
spec = importlib.util.spec_from_file_location("cli_hug", str(Path(__file__).parent.parent / "CLI HUG.py"))
cli_hug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_hug)

ensure_directory = cli_hug.ensure_directory
validate_path = cli_hug.validate_path

def test_validate_path_resolves():
    """Test successful path resolution."""
    path_str = "some_new_folder"
    result = validate_path(path_str)
    assert isinstance(result, Path)
    assert result == Path(path_str).resolve()

def test_validate_path_traversal():
    """Test that path traversal attempts are resolved properly."""
    # Test path resolution without checking existence
    path_str = "base_folder/../target_folder"
    result = validate_path(path_str)
    assert result == Path("target_folder").resolve()

def test_ensure_directory_success(tmp_path):
    """Test successful directory creation."""
    new_dir = tmp_path / "new_folder"
    result = ensure_directory(new_dir)
    assert result is True
    assert new_dir.exists()
    assert new_dir.is_dir()

def test_ensure_directory_already_exists(tmp_path):
    """Test successful creation when directory already exists."""
    existing_dir = tmp_path / "existing_folder"
    existing_dir.mkdir()
    result = ensure_directory(existing_dir)
    assert result is True
    assert existing_dir.exists()
    assert existing_dir.is_dir()

def test_ensure_directory_file_exists(tmp_path):
    """Test when a file already exists at the target path."""
    file_path = tmp_path / "a_file"
    file_path.write_text("dummy content")

    # Path.mkdir will raise FileExistsError if the target exists and is a file
    result = ensure_directory(file_path)

    assert result is False
    assert file_path.exists()
    assert file_path.is_file()

def test_ensure_directory_permission_error(tmp_path, mocker):
    """Test handling of an exception (like PermissionError) during creation."""
    target_dir = tmp_path / "restricted_folder"

    # Mock Path.mkdir to raise PermissionError
    mocker.patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied"))

    result = ensure_directory(target_dir)

    assert result is False
    assert not target_dir.exists()
