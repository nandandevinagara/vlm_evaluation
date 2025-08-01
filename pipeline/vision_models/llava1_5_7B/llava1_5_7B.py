from transformers import pipeline

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf", use_fast=True)


def evaluate(image, prompt):
    messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "image": image},
          {"type": "text", "text": prompt}
      ],
    },
    ]
    out = pipe(text=messages, max_new_tokens=5)
    print(out[0]['generated_text'][1]['content'])

    
if __name__ == "__main__":
    evaluate(image)
