import tqdm
import logging

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

def setup_logger(logger_name, level=logging.INFO):
    l = logging.getLogger(logger_name)
    l.setLevel(level)
    l.addHandler(TqdmLoggingHandler())
    return l