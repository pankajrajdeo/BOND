import os

def configure_runtime() -> None:
    """Set safe defaults for OpenMP to avoid multi-runtime conflicts.

    Applies only if not already set by the environment. Keeps changes minimal
    and portable across Linux, macOS Intel, and Apple Silicon.
    """
    if "KMP_DUPLICATE_LIB_OK" not in os.environ:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"


