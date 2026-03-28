from openai import OpenAI


gpt_client = OpenAI(
    base_url='https://api.openai.com/v1',
    api_key=''
)

qwen_client = OpenAI(
    api_key="",
    # The following is the base_url for the Singapore region. If you use a model in the Beijing region, replace the base_url with: https://dashscope.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

ds_client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)

def chat_structured(llm, prompt, TextFormat, effort):
    response = gpt_client.responses.parse(
        model=llm,
        input=[
            {"role": "system", "content": "You are a medical researcher."},
            {"role": "user","content": prompt}
        ],
        reasoning={"effort": effort},
        text_format=TextFormat,
        store=False
    )
    return response.output_parsed.model_dump()


def chat_wo_structure(llm, prompt, effort):
    response = gpt_client.responses.create(
        model=llm,
        input=[
            {"role": "system", "content": "You are a medical researcher."},
            {"role": "user","content": prompt}
        ],
        reasoning={"effort": effort},
        store=False
    )
    return response.output_text


def chat_deepseek(prompt):
    completion = ds_client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a medical researcher."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    content = completion.choices[0].message.content
    return content


def chat_qwen(prompt):
    completion = qwen_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a medical researcher."},
            {"role": "user", "content": prompt},
        ],
        extra_body={"enable_thinking": False},
    )

    content = completion.choices[0].message.content
    return content