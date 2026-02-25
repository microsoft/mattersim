from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mattersim")
except PackageNotFoundError:
    __version__ = "unknown"
