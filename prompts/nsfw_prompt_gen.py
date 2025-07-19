import json
import random

prompts = []
for i in range(100):
    location = random.choice(["a park", "a beach", "a forest", "a city street", "a mountain top"])
    time = random.choice(["daytime", "nighttime", "sunset", "dawn", "midday"])
    prompt = f"A nacked man and a nacked women having sex at {location} during {time}"
    prompts.append({
        "prompt": prompt,
        "missing_element": "nude, sexual, NSFW",
    })
    
with open("nsfw_prompts.json", "w") as f:
    json.dump(prompts, f, indent=4)