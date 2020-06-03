import json
import logging
import math
import time
from datetime import datetime
from socket import getfqdn

import numpy as np
from tensorrtserver.api import *
import aiohttp
import prometheus_client
import torch
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST
from quart import Quart, request, Response, jsonify
from transformers import InputExample, BertForSequenceClassification

from hitman.server.flask.preprocessing import input_to_vector, get_tokenizer
from hitman.utils.chron import Timer
from hitman.utils.log import setup_prometheus

logger = logging.getLogger(__name__)

prometheus_port = 5000

hostname = getfqdn()

prometheus_registry = None

quart_metrics = {}

model_type = 'bert'
model_name = 'bert-base-uncased'
max_seq_length = 512

pytorch_model = None

tokenizer = get_tokenizer(model_type=model_type, model_name=model_name, do_lower_case=True)


def create_app(config_object="hitman.server.quart.config.Config"):
    app = Quart(__name__)
    app.config.from_object(config_object)

    @app.before_first_request
    async def setup_quart_metrics():
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
            async def metrics():
                return Response(prometheus_client.generate_latest(prometheus_registry), mimetype=CONTENT_TYPE_LATEST)

            logger.info("Quart prometheus metrics endpoint setup done")
            logger.info(prometheus_registry)

            global quart_metrics
            quart_metrics['web_request_count'] = Counter(
                'web_request_count', 'App Request Count',
                ['host', 'app_name', 'method', 'endpoint', 'http_status'],
                registry=prometheus_registry,
            )
            quart_metrics['web_request_latency'] = Histogram('web_request_latency_seconds', 'Request latency',
                                                             ['host', 'app_name', 'endpoint'],
                                                             registry=prometheus_registry,
                                                             )
            quart_metrics['inference_latency'] = Histogram('inference_latency_seconds', 'Inference latency',
                                                           ['host', 'app_name', 'endpoint'],
                                                           registry=prometheus_registry,
                                                           )
            logger.info("Quart metrics registered")
        else:
            logger.warning('No prometheus registry was registered!!!!!')
            setup_prometheus_registry()
            await setup_quart_metrics()

    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    global pytorch_model
    pytorch_model = BertForSequenceClassification.from_pretrained(app.config.get('PYTORCH_MODEL_PATH'))
    logger.info("Pytorch model loaded from {}".format(app.config.get('PYTORCH_MODEL_PATH')))

    return app


quart_app = create_app()


def set_prometheus_port(port):
    global prometheus_port
    prometheus_port = port


def start_timer():
    request.start_time = time.time()


def stop_timer(response):
    resp_time = time.time() - request.start_time
    quart_metrics['web_request_latency'].labels(hostname, 'webapp', request.path).observe(resp_time)
    return response


def record_request_data(response):
    quart_metrics['web_request_count'].labels(hostname, 'webapp', request.method, request.path,
                                              response.status_code).inc()
    return response


async def process_dummy_workload(form):
    t = Timer().start()
    if form['workload_type'] == 'io_bound':
        time.sleep(quart_app.config["DUMMY_REQ_PROC_TIME_SECS"])
    elif form['workload_type'] == 'cpu_bound':
        j = 0
        for i in range(100000):
            j += 1
    elif form['workload_type'] == 'mixed':
        j = 0
        for i in range(100000):
            j += 1
        time.sleep(quart_app.config["DUMMY_REQ_PROC_TIME_SECS"])
    else:
        raise RuntimeError(form['workload_type'] + " workload not supported!")

    data = {'pid': form['pid'],
            'code': 'SUCCESS',
            'req_id': form['req_id'],
            'time': datetime.utcnow(),
            'proc_secs': t.stop()}

    return data



def perform_triton_request(data):
    infer_ctx = InferContext("localhost:9001", ProtocolType.from_str('grpc'), "oic", -1, "", True)
    # prepare inputs
    input_dict = {
        "input_ids": [np.array(data['input_ids'], dtype=np.int64)],
        "input_mask": [np.array(data['attention_mask'], dtype=np.int64)],
        "token_type_ids": [np.array(data['token_type_ids'], dtype=np.int64)]
    }
    # prepare outputs
    output_keys = [
        "output",
    ]
    output_dict = {}
    for k in output_keys:
        output_dict[k] = InferContext.ResultFormat.RAW

    result = infer_ctx.run(input_dict, output_dict, 1)
    logger.info("TensorRT resp: {}".format(result))
    return result

_session = None


async def get_session():
    global _session
    if _session is None:
        _session = aiohttp.ClientSession()
    return _session


async def perform_tensorserve_request(data):
    logger.debug("Tensorserve request! Data: {}".format(data))
    bert_filtered_data = {
        'input_ids': json.dumps(data['input_ids']),
        'attention_mask': json.dumps(data['attention_mask']),
        'token_type_ids': json.dumps(data['token_type_ids'])
    }

    session = await get_session()
    resp = await session.post("http://localhost:8080/predictions/bert_base_test", data=bert_filtered_data)
    logger.info("Content type: {}".format(resp.content_type))
    resp_j = await resp.read()
    resp_j = json.loads(resp_j)
    logger.info("Tensorserve resp: {}".format(resp_j))
    return resp_j


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
    quart_metrics['inference_latency'].labels(hostname, 'webapp', request.path).observe(elapsed_time)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

async def process_real_workload(form):
    t = Timer().start()
    if form['workload_type'] == 'cpu_bound':
        data = query_to_vector(form)
    elif form['workload_type'] == 'mixed':
        data = query_to_vector(form)
        if form['inference'] == 'local':
            perform_local_request(data)
        if form['inference'] == 'trt':
            results = perform_triton_request(data)
            preds = sigmoid_v(results['output'])
            # print(result['output'][0])
            # print(json.dumps(result['output'][0]))
            binary_predictions = np.zeros(preds[0].shape, dtype=int)
            for index, score in enumerate(preds[0]):
                if float(score) > 0.5:
                    binary_predictions[index] = int(1)
            data = json.dumps({'pid': form['pid'],
                'code': 'SUCCESS',
                'req_id': form['req_id'],
                'proc_secs': t.stop(),
                'output': binary_predictions.tolist()})
        else:
            data = await perform_tensorserve_request(data)
            logger.info("DATA {}".format(data))
            data = json.dumps({'torchserve_resp': data})
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


@quart_app.route('/bert_preprocessing', methods=['POST'])
async def bert_tokenizer():
    form = await request.form
    data = await process_dummy_workload(form) if form['is_dummy_workload'] == True else await process_real_workload(
        form)
    jasonified_data = jsonify(data)
    return jasonified_data, 200


@quart_app.errorhandler(500)
def handle_500(error):
    return str(error), 500


if __name__ == '__main__':
    quart_app.run()
