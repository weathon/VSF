from openai import OpenAI
from pydantic import BaseModel
import dotenv
import tqdm

client = OpenAI()

class Prompt(BaseModel):
    prompt: str
    missing_element: str
    question: str


class Prompts(BaseModel):
    prompts: list[Prompt]
    
with open("test_prompt.jsonl", "w") as f:
    for _ in range(4): 
        response = client.responses.parse(
            model="o3",
            input="Write a detailed but not long prompt for generating an image of any object, creature, or scene that is missing a typically expected component or feature. The result should remain visually and physically coherent. It must not create a contradiction or impossibility. For example, acceptable prompts might be: 'a violin without strings', 'a spider with no legs', or 'a staircase with no steps'. (do not repeat these) Do not include self-contradictory cases like 'a crowded dining hall without people' or 'a lit candle with no flame'. The prompt do not need to be physically plausible (i.e. you can have a melted iron without glow). The response should be a descriptive text not a imperative sentences. Put the description and the missing element in the two part of the JSON response. The first value should be the description of the object, creature, or scene, and the second value (just a few words) should be the missing element (e.g. 'cars', 'strings'). The missing element should NOT be mentioned in the first sentence, and put the missing item in 'missing_item' only, do not mention it in prompt. Do NOT repeat the given examples. Each time generate 50 prompts. Additionally, you should give 1, asking if the image fit your description and missing element, for example, if your prompt is 'a violin xyz' and missing element is 'string', the question should be something like 'Is the image showing a violin without strings?'.",
            text_format=Prompts,
            reasoning={"effort": "high"}
        )
        
        f.write(response.output_text + "\n")
        f.flush()
        
