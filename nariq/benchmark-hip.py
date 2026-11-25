"""Run the deterministic 261,632-token text benchmark against llama-server."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18281")
    parser.add_argument("--vram-counter", type=Path)
    args = parser.parse_args()

    def post(path, payload):
        request = Request(
            args.url.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5400) as response:
            return json.load(response)

    def tokenize(content):
        return post("/tokenize", {"content": content, "add_special": False})["tokens"]

    start = tokenize("The code word is COBALT. Read the following records.\n")
    end = tokenize("\nWhat is the code word? Answer with just the word.\nAnswer:")
    records = " ".join(
        f"Station {i} measured {i*i%997} units with no instrument fault."
        for i in range(30000)
    )
    prompt = start + tokenize(records)[:261632 - len(start) - len(end)] + end
    if len(prompt) != 261632:
        raise ValueError(f"Expected 261632 tokens, got {len(prompt)}")
    response = post(
        "/completion",
        {"prompt": prompt, "temperature": 0, "cache_prompt": False, "n_predict": 16},
    )
    result = {
        "timings": response["timings"],
        "recall_passed": "cobalt" in response["content"].lower(),
    }
    if args.vram_counter:
        result["vram_after_gib"] = round(int(args.vram_counter.read_text()) / 2**30, 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
