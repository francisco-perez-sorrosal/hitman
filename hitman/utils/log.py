import distutils.dir_util
import logging
import multiprocessing
import os
import tempfile
from logging.handlers import QueueListener, QueueHandler

from prometheus_client import CollectorRegistry, multiprocess, start_http_server

logger = logging.getLogger(__name__)

LOG_MSG_FORMAT = "%(asctime)s [%(process)d/%(threadName)-12.12s] [%(levelname)-4.4s] -   %(message)s"
LOG_MSG_FORMAT_DEBUG = "%(asctime)s [%(process)d/%(threadName)-12.12s] [%(filename)s:%(lineno)s - %(funcName)20s()] -   %(message)s"
LOG_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def get_log_level_and_formatter(debug):
    log_formatter = logging.Formatter(LOG_MSG_FORMAT, datefmt=LOG_DATE_FORMAT)
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        log_formatter = logging.Formatter(LOG_MSG_FORMAT_DEBUG, datefmt=LOG_DATE_FORMAT)
    else:
        print("Log level {} not recognized. Using info as default log level.".format("info"))
    return log_level, log_formatter


def setup_logging(debug=False):
    log_level, log_formatter = get_log_level_and_formatter(debug)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    # file_handler = logging.FileHandler("{0}/{1}".format(args.output_dir, args.log_file))
    # file_handler.setFormatter(log_formatter)

    # For logging in multiprocessing
    mp_q = multiprocessing.Queue()
    ql = QueueListener(mp_q, console_handler)
    ql.start()

    logging.basicConfig(format=LOG_MSG_FORMAT, datefmt=LOG_DATE_FORMAT, level=log_level, handlers=[console_handler])  # TODO Add file handler if required

    return mp_q


def get_multiprocessor_logger(logging_mp_q, debug = False):
    log_level, log_formatter = get_log_level_and_formatter(debug)
    queue_handler = QueueHandler(logging_mp_q)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(queue_handler)
    return logger


def setup_prometheus(port, is_webserver=False):
    coordination_dir = os.environ.get("prometheus_multiproc_dir", tempfile.gettempdir() + "/prometheus-multiproc-dir/")
    os.environ["prometheus_multiproc_dir"] = coordination_dir
    distutils.dir_util.mkpath(coordination_dir)
    logger.info("Dir for prometheus multiproc created in {}".format(coordination_dir))

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)

    if not is_webserver:
        start_http_server(port, registry=registry)
        logger.info("Prometheus server started on port {}".format(port))

    return registry
