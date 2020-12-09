import logging
import os
import random

import click
import onnx
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    AutoConfig, AutoModelForSequenceClassification,
)
from winmltools.utils import convert_float_to_float16
from winmltools.utils import save_model

print(torch.__version__)

from hitman.utils.log import setup_logging

logger = logging.getLogger(__name__)

global_rng = random.Random()

# MODEL_TYPE: str = 'bert'

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification),
}


class ModelExporter(object):

    def export(self, pytorch_model, input_ids, input_mask, token_type_ids, export_filename):
        raise NotImplementedError("You should call the subclasses")

    def load_exported(self, exported_model_path):
        raise NotImplementedError("You should call the subclasses")


class ONNXExporter(ModelExporter):

    def __init__(self, expected_model_type, output_dir, raw_filename, onnx_opset_version):
        self.expected_model_type = expected_model_type
        self.output_dir = output_dir
        self.raw_filename = raw_filename
        self.onnx_opset_version = onnx_opset_version

    def export(self, pytorch_model, input_ids, input_mask, token_type_ids, exported_filename=None):
        exported_filename = exported_filename if exported_filename else "{}.onnx".format(self.raw_filename)
        model_path = os.path.join(self.output_dir, exported_filename)
        logger.info("Exporting pytorch model in {} to ONNX using opset_version {}".format(model_path, self.onnx_opset_version))
        if self.expected_model_type == "distilbert":
            torch.onnx.export(pytorch_model, (input_ids, input_mask), model_path,
                              input_names=['input_ids', 'input_mask',],
                              output_names=['output'],
                              dynamic_axes={'input_ids': {0: 'batch_size'},  # variable lenght axes
                                            'input_mask': {0: 'batch_size'},  # variable lenght axes
                                            'output': {0: 'batch_size'}},
                              verbose=False,
                              opset_version=self.onnx_opset_version)
        else:
            torch.onnx.export(pytorch_model, (input_ids, input_mask, token_type_ids), model_path,
                              input_names=['input_ids', 'input_mask', 'token_type_ids'],
                              output_names=['output'],
                              dynamic_axes={'input_ids': {0: 'batch_size'},  # variable lenght axes
                                            'input_mask': {0: 'batch_size'},  # variable lenght axes
                                            'token_type_ids': {0: 'batch_size'},  # variable lenght axes
                                            'output': {0: 'batch_size'}},
                              verbose=False,
                              opset_version=self.onnx_opset_version)
        return model_path

    def load_exported(self, exported_model_path):
        logger.info("Loading ONNX model from {}".format(exported_model_path))
        onnx_model = onnx.load(exported_model_path)
        # logger.debug(onnx_model)
        return onnx_model

    def to_fp16(self, input_model_path=None, exported_filename=None):
        if not input_model_path:
            inported_filename = "{}.onnx".format(self.raw_filename)
            input_model_path = os.path.join(self.output_dir, inported_filename)
        onnx_model = self.load_exported(input_model_path)
        logger.info("Converting model {} to fp16!!!".format(input_model_path))
        new_onnx_model = convert_float_to_float16(onnx_model)
        if not exported_filename:
            exported_filename = "{}_fp16.onnx".format(self.raw_filename)
        save_model(new_onnx_model, os.path.join(self.output_dir, exported_filename))


class TorchscriptExporter(ModelExporter):

    def __init__(self, output_dir, raw_filename):
        self.output_dir = output_dir
        self.raw_filename = raw_filename

    def export(self, pytorch_model, input_ids, input_mask, token_type_ids, exported_filename=None):
        exported_filename = exported_filename if exported_filename else "{}.onnx".format(self.raw_filename)
        traced_model_path = os.path.join(self.output_dir, exported_filename)
        logger.info("Exporting pytorch model to Torchscript in {}".format(traced_model_path))
        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing and save it.
        traced_script_module = torch.jit.trace(pytorch_model, [input_ids, input_mask, token_type_ids])
        # traced_script_module = torch.jit.script(model_pytorch(**dummy_input)[1])
        traced_script_module.save(traced_model_path)

    def load_exported(self, exported_model_path):
        raise NotImplementedError("Try later")

    def to_fp16(self, input_model_path=None, exported_filename=None):
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


# def network_to_half(model):
#     """
#     Convert model to half precision in a batchnorm-safe way.
#     """
#     def bn_to_float(module):
#         """
#         BatchNorm layers need parameters in single precision. Find all layers and convert
#         them back to float.
#         """
#         if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#             module.float()
#         for child in module.children():
#             bn_to_float(child)
#         return module
#     return bn_to_float(model.half())


def prepare_dummy_pytorch_inputs(batch_size, seq_length, vocab_size, num_labels, device):
    logger.info("""
    -----------------
    Input preparation
    -----------------""")
    logger.info(f"Batch size {batch_size}")
    logger.info(f"Seq length {seq_length}")
    logger.info(f"Vocab size {seq_length}")
    logger.info(f"Num labels {num_labels}")
    logger.info(f"Device {device}")
    input_ids = ids_tensor([batch_size, seq_length], vocab_size, device=device)
    logger.debug(input_ids.shape)
    input_mask = ids_tensor([batch_size, seq_length], 2, device=device)
    logger.debug(input_mask.shape)
    token_type_ids = ids_tensor([batch_size, seq_length], 2, device=device)
    logger.debug(token_type_ids.shape)
    labels = ids_tensor([batch_size, num_labels], 2, device=device, type=torch.float)
    logger.debug(labels)
    logger.debug(labels.shape)

    return input_ids, input_mask, token_type_ids, labels


@click.command()
@click.option('--debug/--no-debug', default=False)
@click.option('--fp16/--no-fp16', default=False)
@click.option('--device',
              type=click.Choice(['cpu', 'cuda'], case_sensitive=False))
@click.option('--target_format',
              type=click.Choice(['onnx', 'torchscript'], case_sensitive=False))
@click.option('--onnx_opset_version', default=9)
@click.option('--raw_filename', default="model")
@click.option('--example_input_batch_size', default=2)
@click.option('--max_seq_length', default=512)
@click.option('--expected_num_labels', default=245)
@click.option('--expected_model_type', default="bert")
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
def bert_exporter_cli(input_dir,
                      output_dir,
                      target_format,
                      onnx_opset_version,
                      device,
                      example_input_batch_size,
                      max_seq_length,
                      expected_num_labels,
                      raw_filename,
                      expected_model_type,
                      fp16,
                      debug=False):
    setup_logging(debug)

    logger.info(f"Converting a {expected_model_type} model (location: {input_dir}) to {target_format} (location: {output_dir}) Device: {device}")

    model_config, unused_args = AutoConfig.from_pretrained(input_dir, return_unused_kwargs=True)

    assert model_config.model_type == expected_model_type, "Wrong model expectation!"
    assert len(model_config.id2label) == len(model_config.label2id) == expected_num_labels, "Wrong label expectation!"
    assert model_config.__class__.__name__ == MODEL_CLASSES[expected_model_type][0].__name__, \
        f"{model_config.__class__.__name__} vs {MODEL_CLASSES[expected_model_type][0].__name__}"

    logger.info(f"Model configuration loaded:\n {model_config}")
    logger.info(f"Unused args:\n {unused_args}")

    model_pytorch = AutoModelForSequenceClassification.from_pretrained(input_dir, config=model_config,)
    logger.info(model_pytorch)

    assert model_pytorch.__class__.__name__ == MODEL_CLASSES[expected_model_type][1].__name__, \
        f"{model_pytorch.__class__.__name__} vs {MODEL_CLASSES[expected_model_type][1].__name__}"

    assert not model_pytorch.training, "internal error - model should be in eval() mode! "

    model_pytorch.to(device)
    input_ids, input_mask, token_type_ids, _ = prepare_dummy_pytorch_inputs(example_input_batch_size,
                                                                            max_seq_length,
                                                                            model_config.vocab_size,
                                                                            model_config.num_labels,
                                                                            device)

    if expected_model_type == "distilbert":
        dummy_input = {"input_ids": input_ids, "attention_mask": input_mask}
    else:
        dummy_input = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}

    logger.info("Input: {}".format(dummy_input))
    logger.info("Input Ids shape: {}".format(dummy_input['input_ids'].shape))
    dummy_output = model_pytorch(**dummy_input)
    logger.info("Output: {}".format(dummy_output))
    logger.info("Output shape: {}".format(dummy_output[0].shape))
    logger.info("Output shape 0 : {}".format(dummy_output[0][0].shape))
    logger.info("Output shape 1 : {}".format(dummy_output[0][1].shape))

    if target_format == "onnx":
        exporter = ONNXExporter(expected_model_type, output_dir, raw_filename, onnx_opset_version)
    else:
        exporter = TorchscriptExporter(output_dir, raw_filename)

    model_path = exporter.export(model_pytorch, input_ids, input_mask, token_type_ids)
    if target_format == "onnx":
        onnx_model = exporter.load_exported(model_path)
        onnx.checker.check_model(onnx_model)
        logger.info(onnx.helper.printable_graph(onnx_model.graph))
    logger.info(f"Model exported to {model_path}")
    if fp16:
        exporter.to_fp16(input_model_path=model_path)


if __name__ == '__main__':
    bert_exporter_cli()
