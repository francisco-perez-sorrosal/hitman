import numpy as np
from tensorrtserver.api import *

if __name__ == '__main__':

    batch_size = 1
    protocol = ProtocolType.from_str('grpc')

    health_ctx = ServerHealthContext("localhost:9001", protocol, http_headers="", verbose=True)
    print(health_ctx.is_live())

    input_ids = np.random.randint(low=0, high=10000, size=512)
    segment_ids = np.random.randint(low=0, high=1, size=512)
    input_mask = np.random.randint(low=0, high=1, size=512)
    labels = np.random.randint(low=0, high=111, size=112)

    dtype = np.int64
    input_ids = np.array(input_ids, dtype=dtype)[None, ...]  # make bs=1
    input_mask = np.array(input_mask, dtype=dtype)[None, ...]  # make bs=1
    segment_ids = np.array(segment_ids, dtype=dtype)[None, ...]  # make bs=1
    # labels = np.array(input_mask, dtype=dtype)[None, ...]  # make bs=1

    print(segment_ids)
    print(segment_ids.shape)

    # prepare inputs
    input_dict = {
        "input__0": tuple(input_ids[i] for i in range(batch_size)),
        "input__1": tuple(input_mask[i] for i in range(batch_size)),
        "input__2": tuple(segment_ids[i] for i in range(batch_size))
    }

    input_dict = {
        "input_ids": [input_ids],
        "input_mask": [input_mask],
        "token_type_ids": [segment_ids]
    }

    print(input_dict["token_type_ids"])

    # prepare outputs
    output_keys = [
        "output",
        # "output__1"
    ]

    output_dict = {}
    for k in output_keys:
        output_dict[k] = InferContext.ResultFormat.RAW

    infer_ctx = InferContext("localhost:9001", protocol, 'oic', -1, "", True)

    result = infer_ctx.run(input_dict, output_dict, batch_size)

    # get the result
    print(len(result["output"]))
    start_logits = result["output"][0].tolist()
    # end_logits = result["output__1"][0].tolist()

    print(start_logits)
    # print(end_logits)