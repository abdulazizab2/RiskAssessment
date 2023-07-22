import logging

logging.basicConfig(
    filename="logs/logs.log",
    filemode="a",
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
