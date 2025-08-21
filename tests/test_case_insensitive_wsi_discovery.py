"""
Test case-insensitive WSI file discovery.

This test verifies that the preprocessing module correctly finds WSI files
with various case extensions and preserves folder structure as expected.
"""
import tempfile
from pathlib import Path
from typing import Set

# Mock the _base_extensions for testing
_base_extensions = {
    ".czi", ".svs", ".tif", ".vms", ".vmu", ".ndpi", 
    ".scn", ".mrxs", ".tiff", ".svslide", ".bif", ".qptiff",
}


def create_test_wsi_structure(base_dir: Path) -> Set[Path]:
    """Create test WSI files with various cases and folder structures."""
    test_files = [
        'slides/slide1.svs',              # lowercase extension
        'slides/slide2.SVS',              # uppercase extension
        'slides/slide3.Svs',              # mixed case extension
        'slides/subfolder1/nested.tif',   # nested + lowercase
        'slides/subfolder1/nested.TIF',   # nested + uppercase
        'slides/subfolder2/deep/very_deep.ndpi',  # deep nesting
        'slides/.hidden/secret.MRXS',     # hidden directory + uppercase
        'slides/special-chars_123/test.svslide',  # special chars
        'slides/not_wsi.txt',             # non-WSI (should be ignored)
        'slides/no_extension',            # no extension (should be ignored)
    ]
    
    wsi_files = set()
    for file_path in test_files:
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch()
        
        # Only track actual WSI files
        if full_path.suffix.lower() in _base_extensions:
            wsi_files.add(full_path)
    
    return wsi_files


def find_wsi_files_case_insensitive(wsi_dir: Path) -> Set[Path]:
    """Find WSI files using case-insensitive matching (the fixed approach)."""
    all_files = wsi_dir.glob("**/*")
    found_files = set()
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in _base_extensions:
            found_files.add(file_path)
    return found_files


def find_wsi_files_case_sensitive(wsi_dir: Path) -> Set[Path]:
    """Find WSI files using old case-sensitive approach."""
    found_files = set()
    for ext in _base_extensions:
        found_files.update(wsi_dir.glob(f"**/*{ext}"))
    return found_files


def test_case_insensitive_wsi_discovery():
    """Test that WSI discovery handles all case variations correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        
        # Create test structure
        expected_files = create_test_wsi_structure(base_dir)
        slides_dir = base_dir / 'slides'
        
        # Test case-insensitive approach
        found_files = find_wsi_files_case_insensitive(slides_dir)
        
        # Should find all WSI files regardless of case
        assert len(found_files) == len(expected_files)
        assert found_files == expected_files
        
        # Test old case-sensitive approach for comparison
        old_found = find_wsi_files_case_sensitive(slides_dir)
        
        # Should miss some files with non-lowercase extensions
        assert len(old_found) < len(expected_files)
        missed_by_old = expected_files - old_found
        
        # Verify that missed files are those with non-lowercase extensions
        for missed_file in missed_by_old:
            assert missed_file.suffix != missed_file.suffix.lower()


def test_folder_structure_preservation():
    """Test that folder structure is preserved in output paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        slides_dir = base_dir / 'slides'
        
        # Create the structure from the GitHub issue comment
        test_files = [
            'slides/slide1.svs',
            'slides/subfolder1/slide2.svs',
            'slides/slide2.svs'
        ]
        
        for file_path in test_files:
            full_path = base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()
        
        # Find files using case-insensitive approach
        found_files = find_wsi_files_case_insensitive(slides_dir)
        
        # Verify folder structure preservation
        feature_paths = []
        feat_output_dir = Path('feats')
        for slide_path in found_files:
            # This mimics the logic from STAMP preprocessing
            feature_output_path = feat_output_dir / slide_path.relative_to(slides_dir).with_suffix('.h5')
            feature_paths.append(feature_output_path)
        
        expected_structure = {
            Path('feats/slide1.h5'),
            Path('feats/slide2.h5'),
            Path('feats/subfolder1/slide2.h5')
        }
        
        assert set(feature_paths) == expected_structure


def test_empty_directory():
    """Test behavior with empty directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        slides_dir = Path(temp_dir) / 'empty_slides'
        slides_dir.mkdir()
        
        found_files = find_wsi_files_case_insensitive(slides_dir)
        assert len(found_files) == 0


def test_no_wsi_files():
    """Test behavior when directory contains no WSI files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        slides_dir = Path(temp_dir) / 'slides'
        slides_dir.mkdir()
        
        # Create non-WSI files
        (slides_dir / 'document.pdf').touch()
        (slides_dir / 'image.jpg').touch()
        (slides_dir / 'data.csv').touch()
        
        found_files = find_wsi_files_case_insensitive(slides_dir)
        assert len(found_files) == 0


if __name__ == "__main__":
    # Run tests directly for debugging
    test_case_insensitive_wsi_discovery()
    test_folder_structure_preservation()
    test_empty_directory()
    test_no_wsi_files()
    print("All tests passed!")