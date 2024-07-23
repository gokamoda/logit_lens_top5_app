import copy
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware  # 追加
from jinja2 import Environment, FileSystemLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtyping import TensorType
from my_torchtyping import LAYER_PLUS_1, SEQUENCE, HIDDEN_DIM, VOCAB
from typing import Literal

def get_heatmap_colors(
    weight, positive=(7, 47, 97), negative=(103, 0, 31), direction=1
):
    if direction == 1 or direction == 2 and weight >= 0:
        r, g, b = positive
        r = int(r + (255 - r) * (1 - weight))
        g = int(g + (255 - g) * (1 - weight))
        b = int(b + (255 - b) * (1 - weight))
        return f"rgb({r}, {g}, {b})"
    else:
        r, g, b = negative
        r = int(r + (255 - r) * (1 + weight))
        g = int(g + (255 - g) * (1 + weight))
        b = int(b + (255 - b) * (1 + weight))
        return f"rgb({r}, {g}, {b})"

def get_threshold_from_ranking(attention_weights, topk):
    attention_weights = attention_weights.reshape(-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    topk_values, _ = attention_weights.topk(topk)
    return topk_values[-1]

def get_data_for_logit_lens_html(
    tokenizer: AutoTokenizer,
    logits_for_logit_lens: TensorType[LAYER_PLUS_1, SEQUENCE, VOCAB],
    # attention_weights: TensorType[LAYER, HEAD, SEQUENCE, SEQUENCE],
    tokenized_prompt: list[str],
    title: str,
    focus_token: int | None | Literal["final"] = None,
    top_n: int = 5,
):

    # permutate for chartjs
    logits_for_logit_lens = logits_for_logit_lens.permute(1, 0, 2)
    num_tokens, num_layers, _ = logits_for_logit_lens.shape

    # GET TOP N TOKENS
    top_n_tokens = [[{} for _ in range(num_layers)] for _ in range(num_tokens)]
    probs = torch.softmax(logits_for_logit_lens, dim=-1)
    if focus_token is None:
        for token_idx, token_probs in enumerate(probs):
            for layer_idx, layer_probs in enumerate(token_probs):
                top_n_probs, top_n_token_ids = layer_probs.topk(top_n)
                top_n_tokens[token_idx][num_layers - layer_idx - 1] = [
                    {
                        "token": tokenizer.decode([token_id]),
                        "prob": float(prob),
                    }
                    for token_id, prob in zip(top_n_token_ids, top_n_probs)
                ]

    top_1_tokens = [
        [layer[0]["token"] for layer in token] for token in top_n_tokens
    ]

    colors = [
        [get_heatmap_colors(layer[0]["prob"]) for layer in token]
        for token in top_n_tokens
    ]

    callback_str = [
        [
            [f"{pred['prob']:.3f}: {pred['token']}" for pred in layer]
            for layer in token
        ]
        for token in top_n_tokens
    ]

    data_index_info = {}
    data_index_info["logit_lens"] = {}
    data_index_info["logit_lens"]["start"] = 0
    chart_data = {
        "labels": [
            f"Layer {layer_idx}" for layer_idx in reversed(range(num_layers))
        ],
        "tokenized_prompt": tokenized_prompt,
        "datasets": [
            {
                "type": "bar",
                "label": f"Token {token_idx}",
                "data": [1] * num_layers,
                "backgroundColor": colors[token_idx],
                "categoryPercentage": 1,
                "barPercentage": 1,
                "order": 1,
            }
            for token_idx in range(num_tokens)
        ],
    }
    data_index_info["logit_lens"]["end"] = len(chart_data["datasets"]) - 1

    # add attention information
    # take sum of all heads for now
    attention_line_template = [None for _ in range(num_layers)]

    data = {
        "chart_data": chart_data,
        "callback_str": callback_str,
        "top_1_tokens": top_1_tokens,
        "data_index_info": data_index_info,
        "title": title,
    }
    return data

class Web:
    model: AutoModelForCausalLM
    device: str

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.output_dir = Path("latest/logit_lens_web")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}

    def inference(self, prompt):
        inputs = self.tokenizer([prompt], return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                use_cache=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        
        hidden_states = torch.stack(list(output.hidden_states[0]), dim=1).to("cpu")[0]
        sequences = output.sequences.to("cpu")[0]
        tokenized_prompt = [self.tokenizer.decode(i) for i in inputs["input_ids"][0]]
        generated_text = self.tokenizer.decode(sequences)
        logits_for_logit_lens = self.compute_logits_for_logit_lens_from_hidden_states(hidden_states)

        self.write_html(logits_for_logit_lens, tokenized_prompt)

    def compute_logits_for_logit_lens_from_hidden_states(
        self,
        hidden_states: TensorType[LAYER_PLUS_1, SEQUENCE, HIDDEN_DIM],
    ):
        lm_head = self.model.lm_head
        layer_norm = self.model.transformer.ln_f
        num_layers, _, _ = hidden_states.shape

        logits_by_layer = []
        with torch.no_grad():
            for layer_idx, layer_hidden_state in enumerate(hidden_states):
                logits_by_token = []
                for _, token_hidden_state in enumerate(layer_hidden_state):
                    token_hidden_state = token_hidden_state.to(self.device)
                    if layer_idx == num_layers - 1:
                        # last layer
                        y = lm_head(token_hidden_state)
                    else:
                        y = lm_head(layer_norm(token_hidden_state))

                    logits_by_token.append(y)
                logits_by_layer.append(torch.stack(logits_by_token).cpu())
        return torch.stack(logits_by_layer).cpu()
        
    
    def write_html(self, logits, tokenized_prompt):
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("template.html")

        data = {}

        data["logit_lens"] = get_data_for_logit_lens_html(
            tokenizer=self.tokenizer,
            logits_for_logit_lens=logits,
            tokenized_prompt=tokenized_prompt,
            title="Logit Lens",
        )

        output_path = self.output_dir.joinpath("index.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(template.render(data))

model_name = "gpt2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

web = Web(model_name, device)

app = FastAPI()

# CORSを回避するために追加（今回の肝）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # 追記により追加
    allow_methods=["*"],  # 追記により追加
    allow_headers=["*"],  # 追記により追加
)


@app.get("/init")
async def init(token: str):
    web.data[token] = None
    return {"message": "initialized"}

@app.get("/end")
async def end(token: str):
    print("end", token)
    if token in web.data:
        web.data.pop(token)
    print("active sessions", web.data.keys())
    return {"message": "end"}


@app.get("/inference")
async def inference(prompt: str):
    web.inference(prompt)

    html_path = "latest/logit_lens_web/index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    return {"prompt": prompt, "html": html}


@app.get("/value_lens")
async def see_value_lens(token: int, layer: int, head: int):
    print("running lens")
    return value_lens.lens(token=token, layer=layer, head=head)


@app.get("/image")
async def get_image(path):
    return FileResponse(path)


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
