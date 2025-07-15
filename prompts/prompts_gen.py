from openai import OpenAI
from pydantic import BaseModel
import dotenv
import tqdm

client = OpenAI()

class Prompt(BaseModel):
    prompt: str
    missing_element: str

class Prompts(BaseModel):
    prompts: list[Prompt]
    
with open("test_prompt.jsonl", "w") as f:
    response = client.responses.parse(
        model="o3",
        input="Write a detailed but not long prompt for generating an image of any object, creature, or scene that is missing a typically expected component or feature. The result should remain visually and physically coherent. It must not create a contradiction or impossibility. For example, acceptable prompts might be: 'a violin without strings', 'a spider with no legs', or 'a staircase with no steps'. (do not repeat these) Do not include self-contradictory cases like 'a crowded dining hall without people' or 'a lit candle with no flame'. The prompt should be physically plausible and clearly describe both the subject and the missing element. The response should be a descriptive text not a imperative sentences. Put the description and the missing element in the two sentences. The first sentence should be the description of the object, creature, or scene, and the second sentence (just a few words) should be the missing element (e.g. 'cars', 'strings'). The missing element should NOT be mentioned in the first sentence, and put the missing item in 'missing_item' only, do not mention it in prompt. Do NOT repeat the given examples. Each time generate 200 prompts. ",
        text_format=Prompts,
        reasoning={"effort": "high"}
    )
    
    f.write(response.output_text + "\n")
    f.flush()
        
