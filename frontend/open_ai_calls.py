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
    
    # stream = client.chat.completions.create(
    # model="gpt-4o-mini",
    # messages=[{"role": "user", "content": prompt}],
    # stream=True,
    # )
    
    # for response in stream:
    #     if response["message"]["role"] == "assistant":
    #         return response["message"]["content"]["message"]["content"]["text"]
    
    # return "No response from the model."

    completion = client.chat.completions.create(
    model="text-davinci-002",
    store=True,
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ]
    )


# Example messages
messages = [
    "The server is down. Please fix it as soon as possible.",
    "Remember to complete your annual training module by the end of the month.",
]

# Classify each message
for message in messages:
    result = classify_message(message)
    print(f"Message: {message}\nClassification: {result}\n")


if __name__ == "__main__":
    message = input("Enter a message: ")
    result = classify_message(message)
    print(f"Message: {message}\nClassification: {result}\n")