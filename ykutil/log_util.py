import inspect
import logging
import os
from logging import FileHandler


class TruncatingFileHandler(FileHandler):
    def __init__(
        self,
        filename,
        encoding=None,
        delay=False,
        errors=None,
        max_size_mb=10,
        keep_rate=0.5,
    ):
        super().__init__(filename, "a", encoding, delay, errors)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.keep_rate = keep_rate

    def emit(self, record: logging.LogRecord) -> None:
        if self.should_truncate():
            self.truncate()
        return super().emit(record)

    def truncate(self):
        if os.path.exists(self.baseFilename):
            size = os.path.getsize(self.baseFilename)
            delete_first_n_bytes(self.baseFilename, int(size * (1 - self.keep_rate)))

    def should_truncate(self):
        if os.path.exists(self.baseFilename):
            size = os.path.getsize(self.baseFilename)
            return size > self.max_size_bytes
        return False


class RotatingFileHandle:
    """
    A file-like wrapper that automatically rotates log files when they exceed a size limit.
    When the file reaches max_size_mb, it keeps only the most recent portion of the content.

    This implementation uses a pipe to ensure all writes go through our rotation logic,
    preventing subprocess operations from bypassing the size checking.
    """

    def __init__(self, filepath: str, max_size_mb: float | int = 10):
        import threading
        import subprocess

        self.filepath = filepath
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)

        # Create a pipe for communication
        self.read_fd, self.write_fd = os.pipe()

        # Create the actual file
        self.file_handle = open(filepath, "w")
        self.file_handle.close()  # Close it, we'll reopen as needed

        # Start a background thread to read from pipe and write to file
        self._stop_thread = False
        self._thread = threading.Thread(target=self._pipe_reader, daemon=True)
        self._thread.start()

    def _pipe_reader(self):
        """Background thread that reads from pipe and writes to file with rotation logic"""
        import select

        # Use binary mode for the pipe reader to handle all data correctly
        with os.fdopen(self.read_fd, "rb") as pipe_reader:
            while not self._stop_thread:
                # Use select to check if data is available (non-blocking)
                ready, _, _ = select.select([pipe_reader], [], [], 0.1)
                if ready:
                    try:
                        # Read available data in binary mode
                        data_bytes = pipe_reader.read(8192)  # Read in chunks
                        if data_bytes:
                            # Decode to string for writing to text file
                            data = data_bytes.decode("utf-8", errors="ignore")
                            self._write_to_file(data)
                        else:
                            # EOF reached
                            break
                    except Exception as e:
                        # Handle any errors in reading/writing
                        print(f"Error in pipe reader: {e}")
                        break

    def _write_to_file(self, data: str):
        """Write data to file with rotation logic"""
        if not data:
            return

        # Process data in smaller chunks to ensure we don't exceed size limits
        chunk_size = min(1024, self.max_size_bytes // 4)  # Use smaller chunks

        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            if not chunk:
                continue

            # Check if adding this chunk would exceed the limit
            current_size = (
                os.path.getsize(self.filepath) if os.path.exists(self.filepath) else 0
            )
            chunk_size_bytes = len(chunk.encode("utf-8"))

            if current_size + chunk_size_bytes > self.max_size_bytes:
                self._rotate_file()

            # Write chunk to file
            with open(self.filepath, "a") as f:
                f.write(chunk)
                f.flush()

    def write(self, data: str):
        """Write data through the pipe"""
        if not self._stop_thread:
            os.write(self.write_fd, data.encode("utf-8"))

    def _rotate_file(self):
        """Keep the last 50% of the file when rotating"""
        try:
            if not os.path.exists(self.filepath):
                return

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
            try:
                with open(self.filepath, "w") as f:
                    f.write(f"=== LOG ROTATION ERROR: {e} - FILE TRUNCATED ===\n")
            except:
                pass

    def flush(self):
        """Flush the pipe"""
        try:
            os.fsync(self.write_fd)
        except:
            pass

    def close(self):
        """Close the handle and stop background thread"""
        self._stop_thread = True
        try:
            os.close(self.write_fd)
        except:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def fileno(self):
        """Return the write end of the pipe for subprocess to use"""
        return self.write_fd

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False


def delete_first_n_bytes(filepath: str, n: int):
    """
    Delete the first n bytes from a file.

    Args:
        filepath (str): Path to the file to modify
        n (int): Number of bytes to delete from the beginning

    Returns:
        bool: True if successful, False if file doesn't exist or n <= 0
    """
    if n <= 0:
        return False

    if not os.path.exists(filepath):
        return False

    try:
        # Get file size first
        file_size = os.path.getsize(filepath)

        # If n is greater than or equal to file size, truncate the entire file
        if n >= file_size:
            with open(filepath, "w") as f:
                f.write("")
            return True

        # Read the file content after the first n bytes
        with open(filepath, "rb") as f:
            f.seek(n)  # Skip the first n bytes
            remaining_content = f.read()

        # Write the remaining content back to the file
        with open(filepath, "wb") as f:
            f.write(remaining_content)

        return True

    except Exception as e:
        log(f"Error deleting first {n} bytes from {filepath}: {e}", level="ERROR")
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
        fl = TruncatingFileHandler(file_path, max_size_mb=max_bytes)
    else:
        fl = logging.FileHandler(file_path, mode="a")
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
