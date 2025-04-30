try:
    import gigapath
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "gigapath dependencies not installed."
        " Please reinstall stamp using `pip install 'stamp[gigapath]'`"
    ) from e
