import logging
import os
import random

import click
import onnx
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)

print(torch.__version__)

from hitman.utils.log import setup_logging

logger = logging.getLogger(__name__)

global_rng = random.Random()

MODEL_TYPE: str = 'bert'

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),  # Note that before was BertTokenizer
}


def ids_tensor(shape, vocab_size, rng=None, name=None, type=torch.long):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device='cpu').view(shape).contiguous()


def prepare_inputs(batch_size, seq_length, vocab_size, num_labels, type_vocab_size):
    print(type_vocab_size)
    input_ids = ids_tensor([batch_size, seq_length], vocab_size)
    logger.info(input_ids.shape)
    input_mask = ids_tensor([batch_size, seq_length], type_vocab_size)
    logger.info(input_mask.shape)
    token_type_ids = ids_tensor([batch_size, seq_length], type_vocab_size)
    logger.info(token_type_ids.shape)
    labels = ids_tensor([batch_size, num_labels], type_vocab_size)
    logger.info(labels.shape)

    return input_ids, input_mask, token_type_ids, labels


@click.command()
@click.option('--debug/--no-debug', default=False)
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
def exporter_cli(input_dir, output_dir, debug):
    setup_logging(debug)

    logger.info("Exporting pytorch model from {} to {}".format(input_dir, output_dir))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[MODEL_TYPE]
    model_pytorch = model_class.from_pretrained(input_dir)

    model_config = config_class.from_pretrained(
        input_dir,
        num_labels=162,
        finetuning_task="classification",
        cache_dir=None,
        multilabel=True,
    )
    logger.info("Model loaded from {}".format(input_dir))
    input_ids, input_mask, token_type_ids, labels = prepare_inputs(1, 512, model_config.vocab_size, model_config.num_labels, model_config.type_vocab_size)
    dummy_input = {"input_ids": input_ids, "attention_mask": input_mask, "labels": labels}

    logger.info("Input: {}".format(dummy_input))
    dummy_output = model_pytorch(**dummy_input)
    logger.info("Output shape: {}".format(dummy_output))
    logger.info(dummy_output)


    model_path = os.path.join(output_dir, "oic.onnx")
    logger.info("Exporting ONNX model to {}".format(model_path))
    torch.onnx.export(model_pytorch, (input_ids, input_mask, token_type_ids), model_path, input_names=['input_ids', 'input_mask', 'token_type_ids'], output_names=['output'], verbose=True)

    # logger.info("Loading ONNX model from {}".format(model_path))
    # onnx_model = onnx.load(model_path)
    # logger.info(onnx_model)

if __name__ == '__main__':
    exporter_cli()