import json
import logging
import time

import torch
from transformers import AutoModelForSequenceClassification
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class HFTransformersClassificationHandler(BaseHandler):
    """
    HuggingFace Transformers handler class. This handler takes individual requests with serialized
    json arrays with the required BERT inputs, batching them and converting them into Pytorch tensors.
    It returns the probability for each label.
    """

    def __init__(self):
        super(HFTransformersClassificationHandler, self).__init__()
        self.max_seq_length = 512
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read pytorch model from file and put it into evaluation mode in the right device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing code that batches different requests and forms the input tensors for BERT.
        """
        inputs = {
            'input_ids': None,
            'attention_mask': None,
            'token_type_ids': None
        }
        for request in data:
            input_id_tensor = torch.LongTensor(json.loads(request['input_ids'])).to(self.device).reshape(1,
                                                                                                         self.max_seq_length)
            inputs['input_ids'] = input_id_tensor if inputs['input_ids'] == None else torch.cat(
                [inputs['input_ids'], input_id_tensor], 0)
            attention_mask_tensor = torch.LongTensor(json.loads(request['attention_mask'])).to(self.device).reshape(1,
                                                                                                                    self.max_seq_length)
            inputs['attention_mask'] = attention_mask_tensor if inputs['attention_mask'] == None else torch.cat(
                [inputs['attention_mask'], attention_mask_tensor], 0)
            token_type_ids_tensor = torch.LongTensor(json.loads(request['token_type_ids'])).to(self.device).reshape(1,
                                                                                                                    self.max_seq_length)
            inputs['token_type_ids'] = token_type_ids_tensor if inputs['token_type_ids'] == None else torch.cat(
                [inputs['token_type_ids'], token_type_ids_tensor], 0)

        return inputs

    def inference(self, inputs):
        """
        Do the inference on with the corresponding model.
        """
        outputs = self.model(**inputs)
        logits = outputs[0]
        logger.debug("Model predicted this results {}/{}".format(logits, logits.size()))
        preds = logits.detach().cpu().tolist()

        return preds

    def postprocess(self, inference_output):
        # TODO: Predictions post-processing
        return inference_output


bert_torchserve_handler = HFTransformersClassificationHandler()


# Here begins everything...
def handle(raw_data, context):
    try:
        if not bert_torchserve_handler.initialized:
            bert_torchserve_handler.initialize(context)

        if raw_data is None:
            logger.warning("No data was received... so you'll get nothing!")
            return None

        logger.info("Doing inference for {} requests".format(len(raw_data)))
        logger.debug("Raw data: {}".format(raw_data))

        start = time.perf_counter()
        input_data = bert_torchserve_handler.preprocess(raw_data)
        logger.debug("Prepro time {}".format(time.perf_counter() - start))
        output_data = bert_torchserve_handler.inference(input_data)
        post_processed_data = bert_torchserve_handler.postprocess(output_data)
        logger.debug("All time {}".format(time.perf_counter() - start))
        logger.debug("Output data {}".format(output_data))

        return post_processed_data
    except Exception as e:
        raise e
