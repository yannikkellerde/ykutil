import inspect
import logging
import os
from logging.handlers import RotatingFileHandler


class RotatingFileHandle:
    """
    A file-like wrapper that automatically rotates log files when they exceed a size limit.
    When the file reaches max_size_mb, it keeps only the most recent portion of the content.
    """

    def __init__(self, filepath: str, max_size_mb: int = 10):
        self.filepath = filepath
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.handle = open(filepath, "w")

    def write(self, data: str):
        self.handle.write(data)
        self.handle.flush()

        # Check file size and rotate if necessary
        if os.path.getsize(self.filepath) > self.max_size_bytes:
            self._rotate_file()

    def _rotate_file(self):
        """Keep the last 50% of the file when rotating"""
        self.handle.close()

        try:
            # Read the file content
            with open(self.filepath, "r") as f:
                content = f.read()

            # Keep only the last 50% of content (by character count)
            keep_size = len(content) // 2
            new_content = content[-keep_size:] if keep_size > 0 else ""

            # Write back the truncated content
            with open(self.filepath, "w") as f:
                if new_content:
                    f.write("=== LOG ROTATED - OLDER ENTRIES REMOVED ===\n")
                    f.write(new_content)
        except Exception as e:
            # If rotation fails, just truncate the file
            with open(self.filepath, "w") as f:
                f.write(f"=== LOG ROTATION ERROR: {e} - FILE TRUNCATED ===\n")

        # Reopen the file handle
        self.handle = open(self.filepath, "a")

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()

    def fileno(self):
        return self.handle.fileno()

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False


if not "log_util" in logging.Logger.manager.loggerDict:
    logger = logging.getLogger("log_util")
    logger.setLevel(logging.DEBUG)
    hl = logging.StreamHandler()
    hl.setLevel(logging.INFO)
    hl.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(hl)

level_map = {
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "ERROR": logging.ERROR,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
}


def add_file_handler(file_path, level=logging.INFO, max_bytes=None):
    if isinstance(level, str):
        level = level_map[level.upper()]
    if max_bytes:
        fl = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=0, mode="w")
    else:
        fl = logging.FileHandler(file_path, mode="w")
    fl.setLevel(level)
    fl.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fl)


def log(*messages, level=logging.INFO):
    from termcolor import colored

    if type(level) == str:
        level = level_map[level.upper()]
    caller = inspect.stack()[1]
    msg = f'{os.path.basename(caller.filename)}:{caller.lineno} {" ".join(map(str, messages))}'
    if level == logging.ERROR:
        msg = colored(msg, "red")
    elif level == logging.WARNING:
        msg = colored(msg, "yellow")
    logger.log(level, msg)
