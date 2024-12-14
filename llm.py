from huggingface_hub import InferenceClient

def llm_call(message):
    token = ""

    # Please create your own token on HuggingFace and put it in a file called "api_token.txt"
    with open("api_token.txt", "r") as f:
        token = f.read()

    client = InferenceClient(api_key=token)

    messages = [
        { "role": "user", "content": message },
    ]

    stream = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-alpha", 
        messages=messages, 
        temperature=0.5,
        max_tokens=2048,
        top_p=0.7,
        stream=True
    )

    output = ""
    for chunk in stream:
        output += chunk.choices[0].delta.content

    print(output)