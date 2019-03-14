"""Logging support

Use get_module_logger as primary entry into creating a logger object.  Configuration occurs in user profile settings

"""
import os
import sys
import datetime
import logging
import logging.handlers

# File logging constants for FILE_LOGGING_MODE

APPEND_FILE='append'
TIMESTAMPED_FILE='timestamp'
ROTATED_SIZED_FILE='rotatedsized'
DAILY_ROTATED_FILE='dailyrotated'


def establish_logging():
    """Set up a base logger object
    
    Returns:
        logging.Logger: Base logger
    """

    # Importing inline to avoid circular import
    from .config_manager import get_system_profile
    log_settings = get_system_profile().get_settings()['LOGGER_SETTINGS']

    logger = logging.getLogger(log_settings['APP_NAME'])
    if logger.handlers:
        logger.debug("Logger already configured")
        return logger

    formatter = logging.Formatter(log_settings['LOG_FORMAT'])
    handlersList = []

    if log_settings['LOG_TO_FILE']:
        if log_settings['FILE_LOGGING_MODE']==TIMESTAMPED_FILE:
            now=datetime.datetime.now()
            timestamp=now.strftime("%Y_%m_%d_%H_%M_%S")
            logfile="%s_%s.log"%(log_settings['LOGFILE_BASE'],timestamp)
        else:
            logfile="%s.log"%(log_settings['LOGFILE_BASE'])

        folder=os.path.dirname(os.path.realpath(logfile))
        if not os.path.exists(folder):
            os.makedirs(folder)

        if log_settings['FILE_LOGGING_MODE']==TIMESTAMPED_FILE:
            handlersList.append(logging.FileHandler(os.path.expandvars(os.path.expanduser(logfile))))

        elif log_settings['FILE_LOGGING_MODE']==ROTATED_SIZED_FILE:
            handlersList.append(logging.handlers.RotatingFileHandler(logfile, maxBytes=log_settings['ROTATED_SIZED_BYTES'], \
                backupCount=log_settings['ROTATED_SIZED_FILES']))

        elif log_settings['FILE_LOGGING_MODE']==DAILY_ROTATED_FILE:
            handlersList.append(logging.handlers.TimedRotatingFileHandler(logfile, when='d', \
                interval=1, backupCount=log_settings['DAILY_ROTATED_FILES'], encoding=None, delay=False, utc=True))

        else:
            #FILE_LOGGING_MODE by default is APPENDFILE:
            handlersList.append(logging.FileHandler(os.path.expandvars(os.path.expanduser(logfile))))

    if log_settings['LOG_TO_STDOUT']:
        handlersList.append(logging.StreamHandler(sys.stdout))

    if log_settings['LOG_TO_STDERR']:
        handlersList.append(logging.StreamHandler(sys.stderr))

    for hdlr in handlersList:
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    logger.setLevel(log_settings['LOG_LEVEL'])
    return logger

def get_module_logger(module_name="default"):
    """Return a logger for a particular module
    
    Args:
        module_name (str, optional): Name for this module/logger
    
    Returns:
        logging.Logger: Module logger
    """


    establish_logging()

    # Importing inline to avoid circular import
    from .config_manager import get_system_profile
    log_settings = get_system_profile().get_settings()['LOGGER_SETTINGS']

    logger = logging.getLogger("%s.%s"%(log_settings['APP_NAME'],module_name))

    if module_name in log_settings['DEBUG_MODULES']:
        logger.setLevel(logging.DEBUG)
    return logger
