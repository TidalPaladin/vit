try:
    pass
except ImportError as e:
    raise ImportError(
        "transformer_engine is not installed. "
        "Please install it with `pip install --no-build-isolation transformer-engine[pytorch]>=2.0`"
    ) from e
