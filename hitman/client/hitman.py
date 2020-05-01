import asyncio
import csv
import logging
import os
import time
from multiprocessing import Process

import aiohttp
from aiomultiprocess import Pool
from prometheus_client import Counter, Summary, Gauge
from ratelimiter import RateLimiter

from hitman.utils.chron import Timer
from hitman.utils.log import setup_prometheus, get_multiprocessor_logger
from hitman.utils.mp import MPQueue

logger = logging.getLogger(__name__)

prometheus_registry = None


def init_prometeus_registry(port):
    global prometheus_registry  # add this line!
    if prometheus_registry is None:  # see notes below; explicit test for None
        prometheus_registry = setup_prometheus(port)
    else:
        raise RuntimeError("Registry already initialized.")


async def process_request(work):
    endpoint, is_dummy_workload, inference_type, workload_type, data = work

    worker_data.gauge.inc()
    worker_data.counter.labels(direction='out').inc()
    pid = os.getpid()
    worker_data.logger.debug("Process {} sending request {} !".format(str(pid), data['req_id'])) if data[
                                                                                                        'req_id'] % 1000 == 0 else None
    if is_dummy_workload:
        payload = {'is_dummy_workload': True, 'pid': str(pid), 'req_id': str(data['req_id']),
                   'workload_type': workload_type}
    else:
        payload = {'is_dummy_workload': False, 'pid': str(pid), 'req_id': str(data['req_id']),
                   'workload_type': workload_type,
                   'inference': inference_type,
                   'url': data['url'], 'text_a': data['text_a'], 'text_b': data['text_b']}

    timer = Timer().start()
    async with worker_data.session.post(endpoint,
                                        data=payload) as resp:
        resp_j = await resp.json()
        seconds = timer.stop()
        # worker_data.logger.debug(resp_j)
        worker_data.counter.labels(direction='in').inc()
        worker_data.request_summary.observe(seconds)
        worker_data.gauge.dec()

    return resp_j


def rate_limited_iterator(source_data_queue, workload_batch, endpoint, is_dummy_workload, inference_type, workload_type,
                          rate_limiter):
    i = 0
    while i < workload_batch:
        with rate_limiter:
            data = source_data_queue.get()
            yield endpoint, is_dummy_workload, inference_type, workload_type, data
            i += 1


def init_worker(function, debug, logging_mp_q, tcp_conn):
    # bring the magic of logging to multiprocessing workers
    function.logger = get_multiprocessor_logger(logging_mp_q, debug)

    function.logger.info("=========================== Initializing worker ===========================================")
    connector = aiohttp.TCPConnector(limit=tcp_conn, keepalive_timeout=120.0)
    function.session = aiohttp.ClientSession(connector=connector, conn_timeout=120.0, read_timeout=120.0)
    function.logger.info("Worker client sessions with {} TCP connections created".format(tcp_conn))
    function.counter = Counter('client_counter', 'Client counter', ['direction'], registry=prometheus_registry)
    function.gauge = Gauge('client_gauge', 'Client gauge', registry=prometheus_registry, multiprocess_mode='livesum')
    function.request_summary = Summary('client_request_summary', 'Time spent processing a client request',
                                       registry=prometheus_registry)
    function.logger.info("Worker metrics created")
    function.logger.info("===========================================================================================")


def worker_data():
    worker_data.logger
    worker_data.session
    worker_data.rate_limiter
    worker_data.counter
    worker_data.gauge
    worker_data.request_summary


data_queue = MPQueue()


class DataReader(Process):

    def __init__(self, source_datafile):
        super(DataReader, self).__init__()

        self.lines = []
        logger.info("Reading data from file: {}".format(source_datafile))
        with open(source_datafile) as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')
            for row in reader:
                self.lines.append(row)
        self.n_of_lines = len(self.lines)
        logger.info("Rows read: {}".format(self.n_of_lines))

    def run(self) -> None:
        i = 0
        while True:
            logger.debug("Producing request {}".format(i)) if i % 1000 == 0 else None
            data = self.lines[i % self.n_of_lines]
            data['req_id'] = i
            data_queue.put(data)
            if data_queue.full():
                logger.debug("Queue full. Avoid producing elements super fast. Sleeping 15 sec...")
                time.sleep(15)
            i += 1


class MasterClient:

    def __init__(self, master_config, client_config):
        self.master_config = master_config
        self.client_config = client_config

        if not self.client_config.dummy_workload:
            logger.info("Starting data reader process...")
            self.data_reader = DataReader(self.client_config.source_data)
            self.data_reader.start()

        init_prometeus_registry(self.master_config.prometheus_port)

    async def main(self):
        rate_limiter = RateLimiter(max_calls=self.client_config.max_requests_per_sec, period=1.0)

        logger.info("============================================================================================")
        logger.info("CPU count: {}".format(os.cpu_count()))
        logger.info("Workers: {}".format(self.master_config.workers))
        logger.info("Child concurrency: {}".format(self.client_config.child_concurrency))
        logger.info("Dummy workload? {}".format(self.client_config.dummy_workload))
        logger.info("Inference type: {}".format(self.client_config.inference_type))
        logger.info("Workload type: {}".format(self.client_config.workload_type))
        logger.info("Workload batch: {}".format(self.client_config.workload_batch))
        logger.info("Rate limiter set with {} req/sec".format(self.client_config.max_requests_per_sec))
        logger.info("============================================================================================")

        async with Pool(
                processes=self.master_config.workers,
                queuecount=self.master_config.workers,
                childconcurrency=self.client_config.child_concurrency,  # This acts as a rate limiter
                initializer=init_worker,
                initargs=(worker_data, self.master_config.debug,
                          self.master_config.multi_processing_queue,
                          self.master_config.tcp_conn_workers,)
        ) as pool:
            i = 0
            while True:
                results = await pool.map(process_request, rate_limited_iterator(data_queue,
                                                                                self.client_config.workload_batch,
                                                                                self.client_config.endpoint,
                                                                                self.client_config.dummy_workload,
                                                                                self.client_config.inference_type,
                                                                                self.client_config.workload_type,
                                                                                rate_limiter))

                # logger.debug(results)
                i += 1
                logger.info("End of {} batch of {}".format(i, self.client_config.workload_batch))

    def run(self):
        asyncio.run(self.main())
        self.data_reader.join() if self.data_reader else None
