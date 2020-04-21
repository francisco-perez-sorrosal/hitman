import logging

from transformers import (WEIGHTS_NAME,
                          BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer,
                          AlbertConfig,
                          AlbertForSequenceClassification,
                          AlbertTokenizer,
                          )

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}


def get_tokenizer(model_type='bert', model_name='bert-base-uncased', do_lower_case=True, cache_dir=None):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    return tokenizer_class.from_pretrained(model_name,
                                           do_lower_case=do_lower_case,
                                           cache_dir=cache_dir)


def input_to_vector(example,
                    tokenizer,
                    max_length,
                    pad_on_left=False,
                    pad_token=0,
                    pad_token_segment_id=0,
                    mask_padding_with_zero=True):

    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_length,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    logger.debug("*** Example ***")
    logger.debug("guid: %s" % (example.guid))
    logger.debug("Text a: {}\nText b: {}".format(example.text_a, example.text_b))
    logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logger.debug("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
    logger.debug("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

    return input_ids, attention_mask, token_type_ids
