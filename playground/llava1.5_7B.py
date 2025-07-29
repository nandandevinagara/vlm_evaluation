from transformers import pipeline

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")
messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
          #{"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"},
          {"type": "text", "text": "describe the image"},
        ],
    },
]

out = pipe(text=messages, max_new_tokens=20)
print(out)


# WORKS