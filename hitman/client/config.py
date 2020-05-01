import multiprocessing
from dataclasses import dataclass


@dataclass
class MasterClientConfig:
    debug: bool
    prometheus_port: int
    workers: int
    tcp_conn_workers: int
    multi_processing_queue: multiprocessing.Queue


@dataclass
class ClientConfig:
    dummy_workload: bool
    inference_type: str
    workload_type: str
    workload_batch: int
    max_requests_per_sec: int
    endpoint: str
    source_data: str
    child_concurrency: int
