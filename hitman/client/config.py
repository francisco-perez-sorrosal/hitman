import multiprocessing
from dataclasses import dataclass
from prometheus_client import CollectorRegistry

@dataclass
class MasterClientConfig:
    debug: bool
    prometheus_port: int
    workers: int
    tcp_conn_workers: int
    multi_processing_queue: multiprocessing.Queue


@dataclass
class ClientConfig:
    workload_type: str
    workload_batch: int
    max_requests_per_sec: int
    child_concurrency: int
