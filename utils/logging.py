import logging
import os
import time


def exp_logger(
    task: str, t: str, save_path: str, log_level: int = logging.INFO
) -> logging.Logger:
    # Logging
    logger_name = f"{task}_{t}"
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    # Create a handler for writing logs to a file
    log_save_path = os.path.join(save_path, 'logs')
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path, exist_ok=True)
    log_file_name = logger_name + '.log'
    fh = logging.FileHandler(
        os.path.join(log_save_path, log_file_name), encoding='utf-8'
    )
    fh.setLevel(logging.INFO)
    # Create another handler for outputting logs to the console
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # Define the output format for the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def create_child_logger(
    task: str,
    name: str,
    t,
    save_path,
    level=logging.INFO,
    fmt="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
):
    """
    Create a logger.

    Args:
        name (str): The name of the logger.
        log_file (str): The path to the log file.
        level (int): The log level, default is DEBUG.
        fmt (str): The log format, default includes time, name, and level.

    Returns:
        logging.Logger: The configured logger.
    """
    logger_name = f"{task}_{t}_{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a file handler
    log_save_path = os.path.join(save_path, 'logs', logger_name + '.log')
    file_handler = logging.FileHandler(log_save_path, encoding='utf-8')
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def close_logger(logger):
    """
    Close the specified logger.

    Args:
        logger (logging.Logger): The logger to close.
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def setup_logging():
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_format = '%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s'

    # Create handlers for each log level
    info_handler = logging.FileHandler(f'log/{t}_info.log', mode='a')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter(log_format))

    debug_handler = logging.FileHandler(f'log/{t}_debug.log', mode='a')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(log_format))

    warning_handler = logging.FileHandler(f'log/{t}_warning.log', mode='a')
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(logging.Formatter(log_format))

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(
        logging.DEBUG
    )  # Set the root logger's level to the lowest level you want to capture

    # Add the handlers to the root logger
    root_logger.addHandler(info_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(warning_handler)
