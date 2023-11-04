import logging
import sys
import psutil


class Logger:
    def __init__(self):
        file_handler = logging.FileHandler(filename="std.log")
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        handlers = [file_handler, stdout_handler]

        logging.basicConfig(
            handlers=handlers,
            format="%(name)s - %(levelname)s - %(asctime)s -  %(message)s",
            level=logging.DEBUG,
        )

        self.logger = logging.getLogger()

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(
            f"{message} - CPU: {psutil.cpu_percent()}% - RAM: {psutil.virtual_memory().percent}%"
        )

    def warning(self, message):
        self.logger.warning(
            f"{message} - CPU: {psutil.cpu_percent()}% - RAM: {psutil.virtual_memory().percent}%"
        )

    def error(self, message):
        self.logger.error(
            f"{message} - CPU: {psutil.cpu_percent()}% - RAM: {psutil.virtual_memory().percent}%"
        )
