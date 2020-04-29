import logging
import time
from socket import getfqdn

import prometheus_client

from datetime import datetime

import torch
from flask import Flask, request, Response, jsonify
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST
from transformers import InputExample, BertForSequenceClassification

from hitman.server.flask.preprocessing import input_to_vector, get_tokenizer
from hitman.utils.chron import Timer
from hitman.utils.log import setup_prometheus

logger = logging.getLogger(__name__)

prometheus_port = 5000

hostname = getfqdn()

prometheus_registry = None

flask_metrics = {}

model_type = 'bert'
model_name = 'bert-base-uncased'
max_seq_length = 512

pytorch_model = None

tokenizer = get_tokenizer(model_type=model_type, model_name=model_name, do_lower_case=True)


def create_app(config_object="hitman.server.flask.config"):
    app = Flask(__name__)
    app.config.from_object(config_object)

    @app.before_first_request
    def setup_flask_metrics():
        def setup_prometheus_registry():
            global prometheus_registry
            prometheus_registry = setup_prometheus(prometheus_port, is_webserver=True)

        global prometheus_registry
        if prometheus_registry is not None:

            app.before_request(start_timer)
            # The order here matters since we want stop_timer
            # to be executed first
            app.after_request(record_request_data)
            app.after_request(stop_timer)

            @app.route('/metrics')
            def metrics():
                return Response(prometheus_client.generate_latest(prometheus_registry), mimetype=CONTENT_TYPE_LATEST)

            logger.info("Flask prometheus metrics endpoint setup done")
            logger.info(prometheus_registry)

            global flask_metrics
            flask_metrics['web_request_count'] = Counter(
                'web_request_count', 'App Request Count',
                ['host', 'app_name', 'method', 'endpoint', 'http_status'],
                registry=prometheus_registry,
            )
            flask_metrics['web_request_latency'] = Histogram('web_request_latency_seconds', 'Request latency',
                                                             ['host', 'app_name', 'endpoint'],
                                                             registry=prometheus_registry,
                                                             )
            flask_metrics['inference_latency'] = Histogram('inference_latency_seconds', 'Inference latency',
                                                             ['host', 'app_name', 'endpoint'],
                                                             registry=prometheus_registry,
                                                             )
            logger.info("Flask metrics registered")
        else:
            logger.warning('No prometheus registry was registered!!!!!')
            setup_prometheus_registry()
            setup_flask_metrics()

    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    global pytorch_model
    pytorch_model = BertForSequenceClassification.from_pretrained(app.config.get('PYTORCH_MODEL_PATH'))
    logger.info("Pytorch model loaded from {}".format(app.config.get('PYTORCH_MODEL_PATH')))

    return app


flask_app = create_app()


def set_prometheus_port(port):
    global prometheus_port
    prometheus_port = port


def start_timer():
    request.start_time = time.time()


def stop_timer(response):
    resp_time = time.time() - request.start_time
    flask_metrics['web_request_latency'].labels(hostname, 'webapp', request.path).observe(resp_time)
    return response


def record_request_data(response):
    flask_metrics['web_request_count'].labels(hostname, 'webapp', request.method, request.path,
                                              response.status_code).inc()
    return response


def process_dummy_workload(form):
    t = Timer().start()
    if form['workload_type'] == 'io_bound':
        time.sleep(flask_app.config["DUMMY_REQ_PROC_TIME_SECS"])
    elif form['workload_type'] == 'cpu_bound':
        j = 0
        for i in range(100000):
            j += 1
    elif form['workload_type'] == 'mixed':
        j = 0
        for i in range(100000):
            j += 1
        time.sleep(flask_app.config["DUMMY_REQ_PROC_TIME_SECS"])
    else:
        raise RuntimeError(form['workload_type'] + " workload not supported!")

    data = {'pid': form['pid'],
            'code': 'SUCCESS',
            'req_id': form['req_id'],
            'time': datetime.utcnow(),
            'proc_secs': t.stop()}

    return data


def perform_triton_request(data):
    pass


def perform_local_request(data, device='cpu'):
    inputs = {
        'input_ids': torch.LongTensor(data['input_ids']).to(device).reshape(1, max_seq_length),
        'attention_mask': torch.LongTensor(data['attention_mask']).to(device).reshape(1, max_seq_length),
        'token_type_ids': torch.LongTensor(data['token_type_ids']).to(device).reshape(1, max_seq_length)
    }
    start = time.time()
    outputs = pytorch_model(**inputs)
    elapsed_time = time.time() - start
    logger.info("Inference elapsed time {}".format(elapsed_time))
    flask_metrics['inference_latency'].labels(hostname, 'webapp', request.path).observe(elapsed_time)


def process_real_workload(form):
    if form['workload_type'] == 'cpu_bound':
        data = query_to_vector(form)
    elif form['workload_type'] == 'mixed':
        data = query_to_vector(form)
        if form['inference'] == 'local':
            perform_local_request(data)
        else:
            perform_triton_request(data)
    else:
        raise RuntimeError(form['workload_type'] + " workload not supported!")
    return data


def query_to_vector(form):
    t = Timer().start()
    example = InputExample(form['pid'] + "_" + form['req_id'], form['text_a'] + " " + form['url'], form['text_b'])
    input_ids, attention_mask, token_type_ids = input_to_vector(example,
                                                                tokenizer,
                                                                max_length=max_seq_length,
                                                                # pad on the left for xlnet
                                                                pad_on_left=bool(model_type in ['xlnet']),
                                                                pad_token=
                                                                tokenizer.convert_tokens_to_ids(
                                                                    [tokenizer.pad_token])[
                                                                    0],
                                                                pad_token_segment_id=4 if model_type in [
                                                                    'xlnet'] else 0,
                                                                )

    data = {'pid': form['pid'],
            'code': 'SUCCESS',
            'req_id': form['req_id'],
            'time': datetime.utcnow(),
            'proc_secs': t.stop(),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
            }
    return data


@flask_app.route('/bert_preprocessing', methods=['POST'])
def bert_tokenizer():
    form = request.form
    data = process_dummy_workload(form) if form['is_dummy_workload'] == True else process_real_workload(form)
    return jsonify(data), 200


@flask_app.errorhandler(500)
def handle_500(error):
    return str(error), 500


if __name__ == '__main__':
    flask_app.run()
