#!/usr/bin/env python3
"""
Compare GPT‑4o, Gemini‑1.5‑Pro, Claude‑3.5‑Sonnet, and DeepSeek‑VL on ChartQA.

usage:
  python chartqa_multi_eval.py --n 50 --split val
"""

import argparse, base64, csv, io, os, time, json, warnings
from collections import defaultdict
from pathlib import Path

import datasets
from PIL import Image
import re
from rapidfuzz import fuzz
from tqdm.auto import tqdm

# ---------------------------- general helpers ---------------------------- #

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

def fuzzy_equal(a: str, b: str, thresh: float = 90.0) -> bool:
    return fuzz.token_sort_ratio(normalize(a), normalize(b)) >= thresh


def try_parse_num(text: str):
    """Return float if text is numeric, else None."""
    cleaned = re.sub(r"[^\d.\-eE]", "", text)      # keep digits, dot, sign, exponent
    try:
        return float(cleaned)
    except ValueError:
        return None

def answers_match(a: str, b: str, tol: float = 1e-3, fuzz_thresh: float = 90.0) -> bool:
    """
    True if answers are numerically equal within tol OR fuzzy‑string match.
    Handles cases like '9%' vs '9', '1,234' vs '1234', '≈ 42.0' vs '42'.
    """
    a_num, b_num = try_parse_num(a), try_parse_num(b)
    if a_num is not None and b_num is not None:
        return abs(a_num - b_num) <= tol * max(1.0, abs(b_num))
    # else fall back to fuzzy text
    return fuzz.token_sort_ratio(a.lower(), b.lower()) >= fuzz_thresh


def save_csv_row(path: Path, row, header=None):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header and header:
            w.writerow(header)
        w.writerow(row)

def encode_pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ------------------------------- GPT‑4o ---------------------------------- #

from openai import OpenAI, APIError, RateLimitError
os.environ["OPENAI_API_KEY"] = "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"
openai_client = OpenAI()  # uses OPENAI_API_KEY

def ask_gpt4o(img: Image.Image, q: str, model="gpt-4o"):
    img_data_url = "data:image/png;base64," + encode_pil_to_base64_png(img)
    max_retries=10
    for i in range(max_retries):
        try:
            rsp = openai_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"Answer concisely (one word/number):\n{q}"},
                        {"type": "image_url",
                         "image_url": {"url": img_data_url}}
                    ]}],
                max_tokens=32, temperature=0,
            )
            return rsp.choices[0].message.content.strip()
        except (RateLimitError, APIError) as e:
            time.sleep(5 * (2**i))
    return "<error>"

# --------------------------- Gemini 1.5 Pro ------------------------------ #
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


os.environ["GOOGLE_API_KEY"] = "AIzaSyB1mS5FiJfnMN7wKohiVux832VzE57ffEQ"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def ask_gemini(img: Image.Image, q: str):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    image_part = {"mime_type": "image/png", "data": buf.getvalue()}

    prompt = (
    "You are answering a question about a chart.\n"
    "Respond with **one word or one number only** – no explanation.\n\n"
    f"Question: {q}"
   )
    max_retries=10
    for i in range(max_retries):
        try:
            response = gemini_model.generate_content(
                [prompt, image_part],                        # first element is the instruction+question
                generation_config={"temperature": 0, "max_output_tokens": 8},
            )
            return response.text.strip()
        except ResourceExhausted:
            wait = 2 ** i
            print(f"[Gemini] Rate limit hit, retrying in {wait}s...")
            time.sleep(wait)
    return "<rate_limit>"

    


# ---------------------------- Claude 3.5 -------------------------------- #
import anthropic
from anthropic import RateLimitError, APIStatusError

claude_client = anthropic.Anthropic(api_key="YOUR_ANTHROPIC_API_KEY")

def ask_claude(img: Image.Image, q: str,
               model="claude-3.5-sonnet-20240620"):
    img_b64 = encode_pil_to_base64_png(img)
    max_retries=5


    for i in range(max_retries):
        try:
            rsp = claude_client.messages.create(
                model=model, temperature=0, max_tokens=64,
            messages=[{
                "role": "user",
                 "content": [
                    {"type": "text",
                    "text": f"Answer concisely (one word/number):\n{q}"},
                    {"type": "image",
                     "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                 }},
            ]}]
            )
            return rsp.content[0].text.strip()
        except RateLimitError:
            wait = 2 ** i  # exponential backoff
            print(f"[Claude] Rate limit hit. Retrying in {wait}s...")
            time.sleep(wait)
        
        except APIStatusError as e:
            print(f"[Claude] API error: {e}")
            break

    return "<rate_limit>"


"""    
# ----------------------------- DeepSeek‑VL ------------------------------- #

with warnings.catch_warnings():
    warnings.simplefilter("ignore")        # silence HF accelerate warnings
    from transformers import AutoProcessor, AutoModelForVision2Seq

deepseek_name = "deepseek-ai/deepseek-vl-7b-chat"
processor = AutoProcessor.from_pretrained(deepseek_name)
deepseek_model = AutoModelForVision2Seq.from_pretrained(
    deepseek_name, device_map="auto", trust_remote_code=True).eval()

def ask_deepseek(img: Image.Image, q: str):
    prompt = f"<|user|>\n<|image_1|>\n{q}<|end|>\n<|assistant|>"
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to("cuda")
    output = deepseek_model.generate(**inputs, max_new_tokens=64)
    resp = processor.batch_decode(output, skip_special_tokens=True)[0]
    return resp.strip()

"""

# ---------------------------- main evaluation ---------------------------- #

MODEL_FUNCS = {
 #   "gpt-4o": ask_gpt4o,
   "gemini-1.5-pro": ask_gemini,
 #   "claude-3.5-sonnet": ask_claude,
 #   "deepseek-vl": ask_deepseek,
}

def evaluate(n: int, split: str, out_csv: str = "results.csv"):
    ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split=split)
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    header = ["image_id", "model", "question", "llm_answer",
              "ground_truth", "correct"]
    out_path = Path(out_csv); out_path.unlink(missing_ok=True)

    scores = defaultdict(int)  # model -> correct
    for idx, sample in enumerate(tqdm(ds, desc="Samples")):
        img: Image.Image = sample["image"]; q = sample["query"]
        gt = sample["label"][0]; img_id = sample.get("id", idx)

        for name, func in MODEL_FUNCS.items():
            pred = func(img, q)
            correct = answers_match(pred, gt)
            scores[name] += int(correct)

            save_csv_row(out_path,
                         [img_id, name, q, pred, gt, correct],
                         header=header)
            img_name = str(img_id) + ".png"
            img.save(img_name)

    # leaderboard
    print("\n=== Accuracy ===")
    for name in MODEL_FUNCS:
        acc = scores[name] / len(ds) * 100
        print(f"{name:<20}: {acc:5.1f}% {scores[name]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--csv", default="results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    evaluate(args.n, args.split, args.csv)
