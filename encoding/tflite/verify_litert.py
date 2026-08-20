"""Numeric verification of exported models against the real LiteRT runtime.

Not run in CI (needs `pip install ai-edge-litert numpy`). Generate the
model/reference files with a small Go program that calls MarshalFile and
writes the input and tensai.Predict output as raw little-endian float32,
then run: python verify_litert.py
"""
import sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter

def run(name):
    interp = Interpreter(model_path=f"{name}.tflite")
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = np.fromfile(f"{name}_in.bin", dtype=np.float32).reshape(inp["shape"])
    want = np.fromfile(f"{name}_out.bin", dtype=np.float32)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    got = interp.get_tensor(out["index"]).reshape(-1)
    diff = np.max(np.abs(got - want))
    rel = diff / (np.max(np.abs(want)) + 1e-12)
    status = "OK" if rel < 1e-4 else "FAIL"
    print(f"{name}: {status}  max_abs_diff={diff:.3e} rel={rel:.3e}")
    print(f"  litert: {got}")
    print(f"  tensai: {want}")
    return status == "OK"

ok = all([run("mlp"), run("cnn")])
sys.exit(0 if ok else 1)
