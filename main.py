from ollama import chat
from pydantic import BaseModel
from typing import List
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

class Block(BaseModel):
    description: str
    block_number: int
    summarised_data: str

class PetList(BaseModel):
    pets: List[Block]

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_pets_in_image(image) -> PetList:
    image_base64 = image_to_base64(image)
    response = chat(
        model='llama3.2-vision',
        format=PetList.model_json_schema(),
        messages=[
            {
                'role': 'user',
                'content': '''Analyze this flowchart. 
                Each block is connected to a different block.
                For each block, provide:
                - description
                - block number
                Provide summarisation of data provided.
                Summarize and return information for ALL blocks visible in the image.''',
                'images': [image_base64],
            },
        ],
        options={'temperature': 0}
    )
    pets_data = PetList.model_validate_json(response.message.content)
    return pets_data

def analyze_image(image):
    pets_result = analyze_pets_in_image(image)
    result = f"Found {len(pets_result.pets)} blocks in the image:\n"
    for i, block in enumerate(pets_result.pets, 1):
        result += f"\nBlock #{i}:\n{block.model_dump_json(indent=2)}\n"
    return result

iface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Flowchart Analyzer",
    description="Upload an image of a flowchart to analyze the blocks."
)

if __name__ == "__main__":
    iface.launch(share=True)
