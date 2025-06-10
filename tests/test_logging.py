import logging
import asyncio
from ykutil.log_util import TruncatingFileHandler, RotatingFileHandle
import os
from fire import Fire


def test_truncating_file_handler():
    fl = TruncatingFileHandler("test.log", max_size_mb=0.01, keep_rate=0.5)
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fl)
    for i in range(100):
        logger.info("test" * 100)
    assert os.path.exists("test.log")
    print(os.path.getsize("test.log"), fl.max_size_bytes)


async def test_rotating_file_handle():
    """Test that RotatingFileHandle limits file size when used with create_subprocess_shell"""
    log_file = "test_rotating.log"
    max_size_mb = 0.01  # Very small limit for testing

    # Clean up any existing test file
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create RotatingFileHandle
    log_handle = RotatingFileHandle(log_file, max_size_mb=max_size_mb)
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    try:
        # Run a command that generates a lot of output
        # Use a command that will generate more output than our size limit
        command = "python -c \"for i in range(1000): print('A' * 100)\""

        proc = await asyncio.create_subprocess_shell(
            command, stdout=log_handle, stderr=log_handle
        )
        await proc.wait()

        # Flush and check file size
        log_handle.flush()

        assert os.path.exists(log_file), "Log file should exist"
        file_size = os.path.getsize(log_file)
        print(
            f"Final file size: {file_size} bytes, Max allowed: {max_size_bytes} bytes"
        )

        # File size should not exceed max_size_bytes by much (allowing for some overhead from rotation message)
        # We allow some buffer for the rotation message that gets added
        assert (
            file_size <= max_size_bytes + 1000
        ), f"File size {file_size} exceeds limit {max_size_bytes}"

        # Verify the file contains rotation message if it was rotated
        with open(log_file, "r") as f:
            content = f.read()
            if "=== LOG ROTATED - OLDER ENTRIES REMOVED ===" in content:
                print("Log rotation occurred as expected")

    finally:
        log_handle.close()
        # Clean up test file
        if os.path.exists(log_file):
            os.remove(log_file)


if __name__ == "__main__":
    Fire(
        {
            "truncating": test_truncating_file_handler,
            "rotating": lambda: asyncio.run(test_rotating_file_handle()),
        }
    )
