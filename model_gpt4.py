from openai import OpenAI
import os

def load_gpt4_model():
    ## Set the API key and model name
    text_model="gpt-4o"
    OPENAI_API_KEY = 'sk-proj-lrh2hZB45kvHDCmHrmUbT3BlbkFJ3Qa8q2VNkK8o8iMinWlo'
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))
    return client, text_model

def get_instruction(client, model, input_text, max_tokens=30):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides clear and detailed instructions on how to put on a blood pressure cuff. You will give instructions live, so only give instructions for the next 5 seconds."},
        {"role": "user", "content": f"Input: {input_text}\n\nPlease provide a short and direct instruction on what the person should do next to measure the blood pressure based on the input."}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )

    return completion.choices[0].message.content

'''Testing'''
client, model = load_gpt4_model()
instruction1 = get_instruction(client, model, "Please sit", max_tokens=30)
print(instruction1)