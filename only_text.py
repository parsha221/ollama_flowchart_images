from ollama import chat
from pydantic import BaseModel
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

def analyze_image(image) -> str:
    image_base64 = image_to_base64(image)
    response = chat(
        model='llama3.2-vision',
        format=Block.model_json_schema(),
        messages=[
            {
                'role': 'user',
                'content': '''Analyze this image. 
                 Image contains text, extract all text
                 and return the text in the image.''',
                'images': [image_base64],
            },
        ],
        options={'temperature': 0}
    )
    result = response.message.content
    return result

iface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Analyzer",
    description="Upload an image of a flowchart to analyze the image"
)

if __name__ == "__main__":
    iface.launch(share=True)
