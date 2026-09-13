"""LLM clients used across the pipeline.

Credentials are read from the environment:

    export OPENAI_API_KEY=...          # required
    export OPENAI_BASE_URL=...         # optional, for a proxy or gateway

The DeepSeek and Qwen clients are only needed for the cross-model verification
ablation; their keys can be left unset otherwise.
"""

import os
from functools import lru_cache

from openai import OpenAI

SYSTEM_PROMPT = "You are a medical researcher."

# Alibaba Cloud DashScope, Singapore region. For the Beijing region use
# https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _client(key_var, base_url, provider):
    api_key = os.environ.get(key_var)
    if not api_key:
        raise RuntimeError(
            f"{key_var} is not set; it is required to call {provider}."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


@lru_cache(maxsize=None)
def gpt_client():
    return _client(
        "OPENAI_API_KEY",
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "OpenAI",
    )


@lru_cache(maxsize=None)
def qwen_client():
    return _client("QWEN_API_KEY", QWEN_BASE_URL, "Qwen")


@lru_cache(maxsize=None)
def deepseek_client():
    return _client("DEEPSEEK_API_KEY", DEEPSEEK_BASE_URL, "DeepSeek")


def chat_structured(llm, prompt, TextFormat, effort):
    """Schema-constrained call. Returns the parsed object as a dict."""
    response = gpt_client().responses.parse(
        model=llm,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": effort},
        text_format=TextFormat,
        store=False,
    )
    return response.output_parsed.model_dump()


def chat_wo_structure(llm, prompt, effort):
    response = gpt_client().responses.create(
        model=llm,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": effort},
        store=False,
    )
    return response.output_text


def chat_deepseek(prompt):
    completion = deepseek_client().chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return completion.choices[0].message.content


def chat_qwen(prompt):
    completion = qwen_client().chat.completions.create(
        model=os.environ.get("QWEN_MODEL", "qwen-plus"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content
