"""Small path-related helpers shared across the project."""


def path_safe(label: str) -> str:
    """Replace path separators so labels are safe as filename parts."""
    return label.replace("/", "_").replace("\\", "_")
