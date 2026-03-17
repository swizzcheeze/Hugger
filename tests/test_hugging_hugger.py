import pytest
import importlib.util
from pathlib import Path

# Load Hugging Hugger.py module dynamically to bypass the space in the filename
spec = importlib.util.spec_from_file_location("hugging_hugger", str(Path(__file__).parent.parent / "Hugging Hugger.py"))
hugging_hugger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hugging_hugger)

validate_path = hugging_hugger.validate_path

def test_validate_path_success(tmp_path):
    """Test successful directory validation."""
    valid_dir = tmp_path / "valid_folder"
    valid_dir.mkdir()

    result = validate_path(str(valid_dir))

    assert isinstance(result, Path)
    assert result == valid_dir.resolve()

def test_validate_path_not_exists(tmp_path):
    """Test validation fails when directory does not exist."""
    non_existent_dir = tmp_path / "missing_folder"

    with pytest.raises(ValueError, match="Invalid or non-existent path"):
        validate_path(str(non_existent_dir))

def test_validate_path_not_a_directory(tmp_path):
    """Test validation fails when path is a file, not a directory."""
    file_path = tmp_path / "a_file.txt"
    file_path.write_text("dummy")

    with pytest.raises(ValueError, match="Path is not a directory"):
        validate_path(str(file_path))

def test_validate_path_traversal(tmp_path):
    """Test that path traversal attempts are resolved properly."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()

    target_dir = tmp_path / "target"
    target_dir.mkdir()

    # Construct a traversal string from base to target
    traversal_path = str(base_dir / ".." / "target")

    # It should resolve correctly to target_dir if target_dir exists
    result = validate_path(traversal_path)
    assert result == target_dir.resolve()
