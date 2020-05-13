import argparse

import numpy as np
from tensorrtserver.api import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for inference. default: 1')
    parser.add_argument("--server-url", type=str, default="localhost:9001",
                        help="Inference server URL. Default is localhost:9001.")
    parser.add_argument('--format', type=str, required=False, default='onnx',
                        help='Format ("ts"/"onnx") used to ' +
                             'communicate with inference service. Default is "onnx".')
    args = parser.parse_args()

    print("Format: {}".format(args.format))

    batch_size = args.batch_size
    protocol = ProtocolType.from_str('grpc')

    health_ctx = ServerHealthContext(args.server_url, protocol, http_headers="", verbose=True)
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
    if args.format == "ts":
        input_dict = {
            # "input__0": tuple(input_ids[i] for i in range(batch_size)),
            # "input__1": tuple(input_mask[i] for i in range(batch_size)),
            # "input__2": tuple(segment_ids[i] for i in range(batch_size))
            "input__0": [input_ids],
            "input__1": [input_mask],
            "input__2": [segment_ids]
        }
    else:
        input_dict = {
            "input_ids": [input_ids],
            "input_mask": [input_mask],
            "token_type_ids": [segment_ids]
        }


    # prepare outputs

    if args.format == "ts":
        output_keys = [
            "output__0",
        ]
    else:
        output_keys = [
            "output",
        ]

    output_dict = {}
    for k in output_keys:
        output_dict[k] = InferContext.ResultFormat.RAW

    infer_ctx = InferContext(args.server_url, protocol, 'oic', -1, "", True)

    result = infer_ctx.run(input_dict, output_dict, batch_size)

    # get the result
    if args.format == "ts":
        print(len(result["output__0"]))
        start_logits = result["output__0"][0].tolist()
    else:
        print(len(result["output"]))
        start_logits = result["output"][0].tolist()

    print(start_logits)
    # print(end_logits)