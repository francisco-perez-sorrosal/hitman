import asyncio
import logging
import os
from enum import Enum

import aiohttp
from aiomultiprocess import Pool
from prometheus_client import Counter, Summary, Gauge
from ratelimiter import RateLimiter

from hitman.utils.chron import Timer
from hitman.utils.log import setup_prometheus, get_multiprocessor_logger

logger = logging.getLogger(__name__)

prometheus_registry = None


def init_prometeus_registry(port):
    global prometheus_registry  # add this line!
    if prometheus_registry is None:  # see notes below; explicit test for None
        prometheus_registry = setup_prometheus(port)
    else:
        raise RuntimeError("Registry already initialized.")


async def process_request(work):
    endpoint, i, workload_type = work
    worker_data.gauge.inc()
    worker_data.counter.labels(direction='out').inc()
    pid = os.getpid()
    worker_data.logger.debug("Process {} sending request {}!".format(str(pid), str(i)))

    payload = {'pid': str(pid), 'req_id': str(i), 'workload_type': workload_type}
    timer = Timer().start()
    async with worker_data.session.post(endpoint,
                                        data=payload) as resp:
        resp_j = await resp.json()
        seconds = timer.stop()
        worker_data.logger.debug(resp_j)
        worker_data.counter.labels(direction='in').inc()
        worker_data.request_summary.observe(seconds)
        worker_data.gauge.dec()

    return resp_j


def rate_limited_iterator(iterable, rate_limiter):
    for x in iterable:
        with rate_limiter:
            yield x


def init_worker(function, debug, logging_mp_q, tcp_conn):
    # bring the magic of logging to multiprocessing workers
    function.logger = get_multiprocessor_logger(logging_mp_q, debug)

    connector = aiohttp.TCPConnector(limit=tcp_conn, keepalive_timeout=120.0)
    function.session = aiohttp.ClientSession(connector=connector, conn_timeout=120.0, read_timeout=120.0)
    function.logger.info("Worker client sessions with {} TCP connections created".format(tcp_conn))
    function.counter = Counter('client_counter', 'Client counter', ['direction'], registry=prometheus_registry)
    function.gauge = Gauge('client_gauge', 'Client gauge', registry=prometheus_registry, multiprocess_mode='livesum')
    function.request_summary = Summary('client_request_summary', 'Time spent processing a client request', registry=prometheus_registry)
    function.logger.info("Worker metrics created")


def worker_data():
    worker_data.logger
    worker_data.session
    worker_data.rate_limiter
    worker_data.counter
    worker_data.gauge
    worker_data.request_summary


class MasterClient:

    def __init__(self, master_config, client_config):
        self.master_config = master_config
        self.client_config = client_config

        init_prometeus_registry(self.master_config.prometheus_port)

    async def main(self):
        rate_limiter = RateLimiter(max_calls=self.client_config.max_requests_per_sec, period=1.0)
        logger.info("Rate limiter set with {} req/sec".format(self.client_config.max_requests_per_sec))

        async with Pool(
                processes=self.master_config.workers,
                queuecount=self.master_config.workers,
                childconcurrency=self.client_config.child_concurrency,  # This acts as a rate limiter
                initializer=init_worker,
                initargs=(worker_data, self.master_config.debug, self.master_config.multi_processing_queue, self.master_config.tcp_conn_workers,)
        ) as pool:
            logger.info("CPU count: {}".format(os.cpu_count()))
            logger.info("Workload type: {}".format(self.client_config.workload_type))
            loop_idx = 0
            # workload = [("http://localhost:5000/bert_preprocessing", loop_idx + i) for i in range(self.client_config.max_requests_per_sec)]
            while True:
                workload = [("http://localhost:5000/bert_preprocessing",
                             loop_idx + i,
                             self.client_config.workload_type) for i in range(self.client_config.workload_batch)]
                logger.info("Workload batch: {}".format(len(workload)))
                results = await pool.map(process_request, rate_limited_iterator(workload, rate_limiter))
                logger.debug(results)
                loop_idx += self.client_config.max_requests_per_sec

    def run(self):
        asyncio.run(self.main())
