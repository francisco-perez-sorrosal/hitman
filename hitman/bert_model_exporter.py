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


class ModelExporter(object):

    def export(self, pytorch_model, input_ids, input_mask, token_type_ids, export_filename):
        raise NotImplementedError("You should call the subclasses")

    def load_exported(self, exported_model_path):
        raise NotImplementedError("You should call the subclasses")

class ONNXExporter(ModelExporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export(self, pytorch_model, input_ids, input_mask, token_type_ids, export_filename="sample_export_to.onnx"):
        model_path = os.path.join(self.output_dir, export_filename)
        logger.info("Exporting pytorch model to ONNX in {}".format(model_path))
        torch.onnx.export(pytorch_model, (input_ids, input_mask, token_type_ids), model_path,
                          input_names=['input_ids', 'input_mask', 'token_type_ids'], output_names=['output'],
                          verbose=True)

    def load_exported(self, exported_model_path):
        logger.info("Loading ONNX model from {}".format(exported_model_path))
        onnx_model = onnx.load(exported_model_path)
        logger.info(onnx_model)
        return onnx_model


class TorchscriptExporter(ModelExporter):

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export(self, pytorch_model, input_ids, input_mask, token_type_ids, export_filename="traced_model.pt"):
        traced_model_path = os.path.join(self.output_dir, export_filename)
        logger.info("Exporting pytorch model to Torchscript in {}".format(traced_model_path))
        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing and save it.
        traced_script_module = torch.jit.trace(pytorch_model, [input_ids, input_mask, token_type_ids])
        # traced_script_module = torch.jit.script(model_pytorch(**dummy_input)[1])
        traced_script_module.save(traced_model_path)

    def load_exported(self, exported_model_path):
        raise NotImplementedError("Try later")


def ids_tensor(shape, vocab_size, rng=None, name=None, device="cuda", type=torch.long):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=type, device=device).view(shape).contiguous()


def prepare_inputs(batch_size, seq_length, vocab_size, num_labels, type_vocab_size, device):
    input_ids = ids_tensor([batch_size, seq_length], vocab_size, device=device)
    logger.debug(input_ids.shape)
    input_mask = ids_tensor([batch_size, seq_length], type_vocab_size, device=device)
    logger.debug(input_mask.shape)
    token_type_ids = ids_tensor([batch_size, seq_length], type_vocab_size, device=device)
    logger.debug(token_type_ids.shape)
    labels = ids_tensor([batch_size, num_labels], type_vocab_size, device=device)
    logger.debug(labels.shape)

    return input_ids, input_mask, token_type_ids, labels


@click.command()
@click.option('--debug/--no-debug', default=False)
@click.option('--target_format',
              type=click.Choice(['onnx', 'torchscript'], case_sensitive=False))
@click.option('--device',
              type=click.Choice(['cpu', 'cuda'], case_sensitive=False))
@click.option('--num_labels', default=162)
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
def bert_exporter_cli(input_dir, output_dir, target_format, device, num_labels, debug):
    setup_logging(debug)

    logger.info("Exporting pytorch model from {} to {}".format(input_dir, output_dir))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[MODEL_TYPE]
    model_pytorch = model_class.from_pretrained(input_dir)
    assert not model_pytorch.training, "internal error - model should be in eval() mode! "

    model_config = config_class.from_pretrained(
        input_dir,
        num_labels=num_labels,
        finetuning_task="classification",
        cache_dir=None,
        multilabel=True,
        torchscript=True if target_format == "torchscript" else False
    )
    model_pytorch.to(device)
    logger.info("Model loaded from {}".format(input_dir))
    input_ids, input_mask, token_type_ids, labels = prepare_inputs(1, 512, model_config.vocab_size,
                                                                   model_config.num_labels,
                                                                   model_config.type_vocab_size,
                                                                   device)
    dummy_input = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}

    logger.info("Input: {}".format(dummy_input))
    dummy_output = model_pytorch(**dummy_input)
    logger.info("Output shape: {}".format(dummy_output))
    logger.info(dummy_output)

    if target_format == "onnx":
        exporter = ONNXExporter(output_dir)
    else:
        exporter = TorchscriptExporter(output_dir)

    exporter.export(model_pytorch, input_ids, input_mask, token_type_ids)


if __name__ == '__main__':
    bert_exporter_cli()
