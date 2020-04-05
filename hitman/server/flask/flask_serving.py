import logging
import time
from socket import getfqdn

import prometheus_client
from flask import Flask
from flask import request, Response
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST

from hitman.utils.chron import Timer
from hitman.utils.log import setup_prometheus

logger = logging.getLogger(__name__)

prometheus_port = 5000

hostname = getfqdn()

prometheus_registry = None

flask_metrics = {}


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
            logger.info("Flask metrics registered")
        else:
            logger.warning('No prometheus registry was registered!!!!!')
            setup_prometheus_registry()
            setup_flask_metrics()

    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

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
    flask_metrics['web_request_count'].labels(hostname, 'webapp', request.method, request.path, response.status_code).inc()
    return response

from flask import make_response, jsonify
from datetime import datetime

@flask_app.route('/bert_preprocessing', methods=['POST'])
def bert_tokenizer():

    form = request.form

    if form['workload_type'] == 'io_bound':
        time.sleep(flask_app.config["DUMMY_REQ_PROC_TIME_SECS"])
    elif form['workload_type'] == 'cpu_bound':
        t = Timer().start()
        j = 0
        for i in range(100000):
            j += 1
        print(t.stop())
    elif form['workload_type'] == 'mixed':
        t = Timer().start()
        j = 0
        for i in range(100000):
            j += 1
        time.sleep(flask_app.config["DUMMY_REQ_PROC_TIME_SECS"])
        print(t.stop())

    else:
        raise RuntimeError(form['workload_type'] + " workload not supported!")

    data = {'pid': form['pid'],
            'code': 'SUCCESS',
            'req_id': form['req_id'],
            'time': datetime.utcnow()}
    return jsonify(data), 200

@flask_app.errorhandler(500)
def handle_500(error):
    return str(error), 500


if __name__ == '__main__':
    flask_app.run()
