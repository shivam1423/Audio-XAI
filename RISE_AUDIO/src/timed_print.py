import builtins
import sys
import time


_process_start_time = time.perf_counter()
_original_print = builtins.print


def _format_elapsed_prefix() -> str:
    elapsed_seconds = time.perf_counter() - _process_start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"[{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}]"


def _timed_print(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
    if file is None:
        file = sys.stdout
    prefix = _format_elapsed_prefix()
    _original_print(prefix, *args, sep=sep, end=end, file=file, flush=flush)


# Install the timed print globally upon import
builtins.print = _timed_print


