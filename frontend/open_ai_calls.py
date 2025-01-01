from openai import OpenAI

open_ai_key = "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"

client = OpenAI(
    api_key=open_ai_key,
)

def classify_message(message):
    prompt = f"""
    Classify the following workplace message as "urgent" or "not urgent. Provide response as number ONLY, 1 for urgent and 0 for not urgent." 
    Message: "{message}"
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    )

    print(completion.choices[0].message)
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")
            # return chunk.choices[0].delta.content   
        
    # return "No response"


if __name__ == "__main__":
    message = input("Enter a message: ")
    result = classify_message(message)
    print(f"Message: {message}\nClassification: {result}\n")