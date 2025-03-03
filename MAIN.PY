from ollama import chat
from pydantic import BaseModel
from typing import List
import gradio as gr
import base64
from io import BytesIO
from PIL import Image
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain


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

def convert_text_to_embeddings(text, user_question):
    text_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=30000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    text_chunks.extend(chunks)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_texts(text_chunks, embeddings)
    match = vector_store.similarity_search(user_question)
    return {"match": match}

def analyze_and_convert(image, user_question):
    text = analyze_image(image)
    embeddings_result = convert_text_to_embeddings(text, user_question)
    
    # Extract the plain text content of the matched chunks
    #matched_chunks = [doc.page_content for doc in embeddings_result["match"]]
    
    # Load the LLM and QA chain
    llm = OllamaLLM(model="llama3.1")
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Generate the response
    response = chain.run(input_documents=embeddings_result["match"], question=user_question)

    result = f"Extracted_text: {text}\n\n"
    result += f"User Question: {user_question}\n\n"
    result += "Analysis Result:\n"
    result += f"{response}\n"
    return result

iface = gr.Interface(
    fn=analyze_and_convert,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Type Your Question Here")
    ],
    outputs="text",
    title="Flowchart Analyzer",
    description="Upload an image of a flowchart to analyze the blocks."
)

if __name__ == "__main__":
    iface.launch(share=True)
