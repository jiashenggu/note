## uvicorn log问题
uvicorn的python启动需要添加init_logger来阻止uvicorn替换logger config

[source](https://segmentfault.com/q/1010000042109567?_ea=252452454)

## log config
```

import logging
import sys
import socket
import os
import datetime
import socket

def get_private_ip():
    # The socket connects to a Google's DNS server IP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

if "BASE_MODEL_PATH" in os.environ:
    base_model_path = os.environ["BASE_MODEL_PATH"]
else:
    base_model_path = "......"
base_model_name = base_model_path.split("/")[-1].split(".")[0]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s"
)
private_ip = get_private_ip()
if "LOG_NAME" in os.environ:
    log_name = os.environ["LOG_NAME"]
else:
    log_name = "dev"
log_file_name = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"logs/{base_model_name}_{log_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{private_ip}.log",
)
os.makedirs(os.path.dirname(log_file_name), exist_ok=True)

file_handler = logging.FileHandler(log_file_name, mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

    def isatty(self):
        return False


# Redirect standard output and standard error to the log file
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)
```
