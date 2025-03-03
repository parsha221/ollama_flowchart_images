from ollama import chat
from pydantic import BaseModel
from typing import List
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

class Block(BaseModel):
    summarised_data: str

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_text_in_image(image) -> str:
    try:
        image_base64 = image_to_base64(image)
        response = chat(
            model='llama3.2-vision',
            messages=[
                {
                    'role': 'user',
                    'content': '''Analyze this image.
                    Image contains text, extract all text
                    and return text in the image.''',
                    'images': [image_base64],
                },
            ],
            options={'temperature': 0}
        )
        text_data = response.message.content
        return text_data
    except Exception as e:
        return f"An error occurred: {e}"

def analyze_image(image):
    text_result = analyze_text_in_image(image)
    result = text_result
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
