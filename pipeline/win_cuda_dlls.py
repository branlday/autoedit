"""On Windows, pip-installed NVIDIA wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12,
etc.) drop their DLLs inside site-packages rather than somewhere Windows'
DLL loader searches by default. ctranslate2 (which faster-whisper relies on)
then fails with errors like:

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

even though the file is present on disk. This registers those directories
with os.add_dll_directory() at startup so the OS loader can find them,
without requiring any manual PATH edits.

No-op on non-Windows platforms.
"""
import os
import sys
import glob
import importlib.util


def register_nvidia_dll_dirs():
    if sys.platform != "win32":
        return

    registered = []

    def _register(bin_dir):
        if not os.path.isdir(bin_dir):
            return
        # ctranslate2's native loader on Windows only consults the PATH
        # environment variable - it does NOT use the os.add_dll_directory()
        # extended search list, even though that's the "modern" API for
        # this. So PATH is the one that actually matters; add_dll_directory
        # is included too since it's harmless and helps other DLL loaders.
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                pass
        registered.append(bin_dir)

    for pkg_name in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc", "nvidia.cuda_runtime"):
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or not spec.submodule_search_locations:
            continue
        for base_dir in spec.submodule_search_locations:
            _register(os.path.join(base_dir, "bin"))

    # Fallback: some versions lay DLLs out slightly differently - do a
    # broader search under any installed "nvidia" namespace package.
    if not registered:
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            for base_dir in spec.submodule_search_locations:
                for bin_dir in glob.glob(os.path.join(base_dir, "*", "bin")):
                    _register(bin_dir)

    if registered:
        print(f"Added {len(registered)} NVIDIA DLL director{'y' if len(registered)==1 else 'ies'} to PATH for CUDA loading.")
