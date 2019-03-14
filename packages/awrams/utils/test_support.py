import logging

class MockLoggingHandler(logging.Handler):
    """Mock logging handler to check for expected logs.
    From http://stackoverflow.com/a/1049375
    """

    def __init__(self, *args, **kwargs):
        self.reset()
        logging.Handler.__init__(self, *args, **kwargs)

    def emit(self, record):
        self.messages[record.levelname.lower()].append(record.getMessage())

    def reset(self):
        self.messages = {
            'debug': [],
            'info': [],
            'warning': [],
            'error': [],
            'critical': [],
        }

    def _add_message(self,level,content):
        self.messages[level.lower()].append(content)

def between(val,x,y):
    x,y = min(x,y),max(x,y)
    return val >= x and val <= y

