"""Generate testdata for TestAgainstHF from the Python tokenizers library.

Not run in CI (needs `pip install tokenizers` and network access). It
downloads the real GPT-2 and Qwen2.5 tokenizer.json files, encodes a
corpus that stresses every scanner branch with the reference
implementation, and writes both into testdata/; the Go test then compares
byte-level BPE, pre-tokenization, special tokens, and decode round-trips
against it. Run: python verify_hf.py && go test .
"""
import json
import os
import urllib.request

from tokenizers import Tokenizer

MODELS = {
    "gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
    "qwen": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
    # llama-architecture model with a Digits(individual_digits) pre-tokenizer
    "smollm2": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json",
}

CORPUS = [
    "Hello, I'm a language model,",
    "The quick brown fox jumps over the lazy dog.",
    "I'LL and I'll and it's and IT'S and won't and WON'T",
    "  leading spaces and   multiple   gaps  ",
    "tabs\tand\nnewlines\r\nmixed \n\n whitespace \t\n x",
    "numbers 123 4567 89 0 12345678901234567890",
    "punct!!! ...---??? ***(())[]{}",
    "mixed42text99with7digits",
    "日本語のテキストと English が混ざる場合。",
    "emoji 🎉🎊 and symbols ©®™ αβγ",
    "trailing newline\n",
    "\n",
    "   ",
    "a",
    "'s",
    "code: if (x != 0) { return x*2; } // comment",
    "quotes \"double\" and 'single' and `backtick`",
    "hy-phen-ated and snake_case and camelCase",
    "spaces before punct , and . here !",
    "\r\ncrlf\r\n\r\nparagraphs\r\n",
    "München Zürich naïve café résumé",
    "ends with spaces   ",
    "<|endoftext|> in the middle <|endoftext|>",
    "digit torture: 1 22 333 a1 1a  5  x 6.7.8, ９８７ ①②③ 42nd",
    " 1", "  1", "a  42", "abc123def", "3.14159", "\n5\t6 7",
]

os.makedirs("testdata", exist_ok=True)
for name, url in MODELS.items():
    path = f"testdata/{name}.json"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    tok = Tokenizer.from_file(path)
    refs = [tok.encode(s).ids for s in CORPUS]
    json.dump({"corpus": CORPUS, "ids": refs}, open(f"testdata/ref_{name}.json", "w"))
    print(name, "ok")

# Random-string fuzz reference for the Digits pre-tokenizer (TestSmolLM2Fuzz).
import random

random.seed(9)
tok = Tokenizer.from_file("testdata/smollm2.json")
alphabet = list("abz ABZ  019９①\t\n.,'!-_日本語é🎉'sTREx") + ["'ll", "42", " 7", "  88 "]
corpus = ["".join(random.choice(alphabet) for _ in range(random.randint(1, 60))) for _ in range(2000)]
json.dump({"corpus": corpus, "ids": [tok.encode(s).ids for s in corpus]},
          open("testdata/ref_smollm2_fuzz.json", "w"))
print("smollm2 fuzz ok")
