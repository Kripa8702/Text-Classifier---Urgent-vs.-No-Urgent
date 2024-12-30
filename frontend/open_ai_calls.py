from openai import OpenAI

open_ai_key = "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"

client = OpenAI(
    api_key=open_ai_key,
)

def classify_message(message):
    prompt = f"""
    Classify the following workplace message as "urgent" or "not urgent." 
    Message: "{message}"
    Response:
    """
    
    stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
    )
    
    for response in stream:
        if response["message"]["role"] == "assistant":
            return response["message"]["content"]["message"]["content"]["text"]
    
    return "No response from the model."


# Example messages
messages = [
    "The server is down. Please fix it as soon as possible.",
    "Remember to complete your annual training module by the end of the month.",
]

# Classify each message
for message in messages:
    result = classify_message(message)
    print(f"Message: {message}\nClassification: {result}\n")