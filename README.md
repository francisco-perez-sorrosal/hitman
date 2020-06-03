# Hitman

[![Build Status](https://travis-ci.org/francisco-perez-sorrosal/hitman.svg?branch=master)](https://travis-ci.org/francisco-perez-sorrosal/hitman)

```shell script
pip install -e .
```

```shell script
mkdir -p /tmp/webapp_multiproc /tmp/clients_multiproc
docker-compose up -d
```

```shell script
# Warning
ulimit -S -n 2048
```

# Server

```shell script
prometheus_multiproc_dir=/tmp/webapp_multiproc hitman_cli --workers 1 --tcp_conn_workers 100 server --framework flask
```

## Gunicorn
```shell script
prometheus_multiproc_dir=/tmp/webapp_multiproc gunicorn -c hitman/gunicorn_conf.py -w 1 -b 127.0.0.1:5000 --log-level=debug -k gevent --worker-connections 2000 --threads 1 --timeout 60 hitman.__main__:flask_app
```

# Client

```shell script
prometheus_multiproc_dir=/tmp/clients_multiproc hitman_cli --workers 1 --tcp_conn_workers 100 client --workload_type io_bound --workload_batch 100 --max_requests_per_sec 1
```

# Interface

```shell script
http://localhost:9090/
http://localhost:8000/metrics
http://localhost:5000/metrics
```

```shell script
# Prometheus queries
client_gauge{job="hitman"}
rate(web_request_latency_seconds_bucket{endpoint="/bert_preprocessing", job="hitman-flask"}[5m])
rate(client_counter_total[1m])
rate(web_request_latency_seconds_count{endpoint="/bert_preprocessing"}[1m])
rate(client_counter_total{direction="out"}[1m])
rate(client_counter_total{direction="in"}[1m])


# How many seconds take the 95 of requests to be processed in the last 5 mins(Percentile 95)
histogram_quantile(0.95, sum(rate(web_request_latency_seconds_bucket{endpoint="/bert_preprocessing"}[5m])) by (le))

# Percentage of request served in 1 second in the last 5 mins
sum(rate(web_request_latency_seconds_bucket{endpoint="/bert_preprocessing", le="1.0"}[5m])) by (job)
```


# Average time in GPU queue
rate(nv_inference_queue_duration_us[1m])/(rate(nv_inference_request_success[1m])+rate(nv_inference_request_failure[1m]))*0.001


# Requests per second at Tensor RT
nv_inference_request_success/nv_inference_request_duration_us/0.000001
# Inference per second at Tensor RT
nv_inference_count/nv_inference_compute_duration_us/0.000001

```shell script
siege -c 200 -r 2 http://localhost:5000/bert_preprocessing\?id="x"
```

```shell script
prometheus_multiproc_dir=/tmp/clients_multiproc hitman_cli --workers 10 --tcp_conn_workers 120 client --child_concurrency 120 --max_requests_per_sec 10000
prometheus_multiproc_dir=/tmp/webapp_multiproc gunicorn -c hitman/gunicorn_conf.py -w 2 -b 127.0.0.1:5000 -k gevent --worker-connections 2048 --threads 1 --timeout 5 --keep-alive 5 --backlog 4096  hitman.__main__:flask_app
```

# Quart
```shell script
prometheus_multiproc_dir=/tmp/webapp_multiproc hitman_cli --debug --workers 1 --tcp_conn_workers 100 server --framework quart
prometheus_multiproc_dir=/tmp/webapp_multiproc gunicorn -c hitman/gunicorn_conf.py -w 16 -b 127.0.0.1:5000 -k uvicorn.workers.UvicornWorker --worker-connections 2048 --threads 8 --timeout 30 --keep-alive 30 --backlog 4096  --preload hitman.__main__:quart_app
```
 

# Tensor RT
docker pull nvcr.io/nvidia/tensorrt:20.03-py3

sudo docker network create inference_network
sudo docker run --gpus '"device=0"' --rm -p8000:8000 --shm-size=1g --ulimit  memlock=-1 --ulimit stack=67108864 --net inference_network --network-alias=trt_server -v/home/fperez/dev/models/tensorrt:/models nvcr.io/nvidia/tensorrt:20.03-py3 giexec --onnx=/models/bert-onnx/test/oic.onnx --device=0 --safe

```
# Get server health
curl -v localhost:9000/api/health/live
curl -v localhost:9000/api/health/ready

# Get current models' APIs
curl -v localhost:9000/api/status |more
```

# Several configs

Single request/sec (Test mode)
prometheus_multiproc_dir=/tmp/clients_multiproc hitman_cli --workers 1 --tcp_conn_workers 400 --debug client --workload_type mixed --child_concurrency 75 --workload_batch 1 --max_requests_per_sec 1
prometheus_multiproc_dir=/tmp/webapp_multiproc gunicorn -c hitman/gunicorn_conf.py -w 1 -b 127.0.0.1:5000 --log-level=debug -k gevent --worker-connections 150 --threads 1 --timeout 30 --keep-alive 30 --backlog 4096  --preload hitman.__main__:flask_app

1K Q/sec
prometheus_multiproc_dir=/tmp/webapp_multiproc gunicorn -c hitman/gunicorn_conf.py -w 4 -b 127.0.0.1:5000 -k gevent --worker-connections 2048 --threads 1 --timeout 30 --keep-alive 30 --backlog 4096  hitman.__main__:flask_app
prometheus_multiproc_dir=/tmp/clients_multiproc hitman_cli --workers 2 --tcp_conn_workers 400 --debug client --workload_type cpu_bound --child_concurrency 75 --workload_batch 425 --max_requests_per_sec 1024

1.5K Q/sec
prometheus_multiproc_dir=/tmp/webapp_multiproc gunicorn -c hitman/gunicorn_conf.py -w 16 -b 127.0.0.1:5000 -k gthread --worker-connections 2048 --threads 8 --timeout 30 --keep-alive 30 --backlog 4096  --preload hitman.__main__:flask_app
prometheus_multiproc_dir=/tmp/clients_multiproc hitman_cli --workers 8 --tcp_conn_workers 200 --debug client --workload_type cpu_bound --child_concurrency 50 --workload_batch 2000 --max_requests_per_sec 2048


# Torch serve

#### Point to the BERT files and create .mar
torch-model-archiver --model-name bert_base_test --version 1.0 --serialized-file /Users/fperez/dev/models/pytorch_model.bin --extra-files /Users/fperez/dev/models/vocab.txt,/Users/fperez/dev/models/config.json --handler /Users/fperez/dev/hitman/hitman/torch_serve_handler.py
mv bert_base_test.mar ~/dev/model_store

#### Start the server
torchserve --start --ts-config ~/dev/hitman/resources/config.properties --model-store ~/dev/model_store --models all

#### Change config a bit
curl -v -X POST "localhost:8081/models?url=bert_base_test.mar&batch_size=8&max_batch_delay=1000&initial_workers=1"

curl -v -X PUT "localhost:8081/models/bert_base_test?max_workers=1"
#### Check config
curl localhost:8081/models/bert_base_test

#### Start the server
torchserve --stop

#### Note arrays to work should be 512 elements (BERT's max_seq_len)
curl --header "Content-Type: application/json" -X POST localhost:8080/predictions/bert_base_test --data '{"input_ids": [101,9033,2290], "attention_mask": [1,1,1], "token_type_ids": [1,1,1]}'

curl --header "Content-Type: application/json" -X POST localhost:9000/api/infer/oic --data '{"input_ids": [101,9033,2290], "attention_mask": [1,1,1], "token_type_ids": [1,1,1]}'


prometheus_multiproc_dir=/tmp/webapp_multiproc gunicorn -c hitman/gunicorn_conf.py -w 145 -b 127.0.0.1:5000 -k gthread --worker-connections 2048 --threads=8 --timeout 90 --keep-alive 30 --backlog 4096  --preload hitman.__main__:flask_app
prometheus_multiproc_dir=/tmp/clients_multiproc hitman_cli --workers 8 --tcp_conn_workers 800  client --workload_type mixed --child_concurrency 8 --workload_batch 8192 --max_requests_per_sec 768 --inference_type trt