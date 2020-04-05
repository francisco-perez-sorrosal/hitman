import dataclasses
import logging
import math

import click

from hitman.client.config import MasterClientConfig, ClientConfig
from hitman.client.hitman import MasterClient
from hitman.server import flask_app
from hitman.utils.log import setup_logging

logger = logging.getLogger(__name__)

CONTEXT_OPTIONS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_OPTIONS)
@click.option('--workers', default=1, help='the number of workers to run')
@click.option('--tcp_conn_workers', default=20, help='the number of tcp connections per worker')
@click.option('--prometheus_port', default=8000)
@click.option('--debug/--no-debug', default=False,
              envvar='REPO_DEBUG')
@click.version_option(version='1.0.0')
@click.pass_context
def hitman_cli(ctx, debug, prometheus_port, workers, tcp_conn_workers):
    logging_mp_q = setup_logging(debug)
    ctx.obj = {'debug': debug,
               'prometheus_port': prometheus_port,
               'workers': workers,
               'tcp_conn_workers': tcp_conn_workers,
               'multi_processing_queue': logging_mp_q}


def dataclass_from_dict(data_class, dictionary):
    fieldtypes = {f.name: f.type for f in dataclasses.fields(data_class)}
    try:
        # return data_class(**{f: dataclass_from_dict(fieldtypes[f], dictionary[f]) for f in dictionary})
        return data_class(**{f: dictionary[f] for f in dictionary})
    except Exception as e:
        logger.error(e)
        return dictionary  # Not a dataclass field


@hitman_cli.command()
@click.option('--workload_type',
              type=click.Choice(['cpu_bound', 'io_bound', 'mixed'], case_sensitive=False))
@click.option('--workload_batch', default=100, help='requests in each batch that will be consumed by workers')
@click.option('--max_requests_per_sec', default=math.inf, help='max requests/sec')
@click.option('--child_concurrency', default=25, help='child_concurrency')
@click.pass_context
def client(ctx, **kwargs):
    if kwargs['workload_type'] is None:
        kwargs['workload_type'] = 'io_bound'
    master_config = dataclass_from_dict(data_class=MasterClientConfig, dictionary=ctx.obj)
    client_config = dataclass_from_dict(data_class=ClientConfig, dictionary=kwargs)

    master_client = MasterClient(master_config, client_config)
    master_client.run()


@hitman_cli.command()
@click.option('--framework',
              type=click.Choice(['flask', 'cherry'], case_sensitive=False))
@click.pass_context
def server(ctx, **kwargs):
    if kwargs['framework'] == 'flask':
        workers = ctx.obj['workers']
        logger.info("Workers: {}".format(workers))
        # options = {  # These cause weird effects
        #     'threaded': False,
        #     'processes': workers
        # }
        flask_app.run(debug=True)
    else:
        logger.warning("GUNICORN Server not implemented yet")


if __name__ == '__main__':
    hitman_cli()
