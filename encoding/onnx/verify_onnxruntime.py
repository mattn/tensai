"""Numeric verification of exported models against onnxruntime.

Not run in CI (needs `pip install onnx onnxruntime numpy`). Generate the
model/reference files with a small Go program that calls MarshalFile and
writes the input and tensai.Predict output as JSON {"input": [...],
"output": [...]}, then run: python verify_onnxruntime.py name (1,C,H,W)
"""
import ast
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort


def run(name, shape):
    model = onnx.load(f"{name}.onnx")
    onnx.checker.check_model(model)
    ref = json.load(open(f"{name}.json"))
    x = np.array(ref["input"], dtype=np.float32).reshape(shape)
    sess = ort.InferenceSession(f"{name}.onnx", providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x})[0].reshape(-1)
    want = np.array(ref["output"], dtype=np.float32)
    err = np.max(np.abs(out - want) / (1 + np.abs(want)))
    print(f"{name}: checker OK, max rel err vs tensai Predict = {err:.2e}")
    assert err < 1e-5, (out[:5], want[:5])


if __name__ == "__main__":
    run(sys.argv[1], ast.literal_eval(sys.argv[2]))
    print("onnxruntime output matches tensai")
