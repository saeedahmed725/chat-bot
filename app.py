import os
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDu6JN_L9gojotvFa8ALFgYO3mux9eB3-U'

from chat_memory import app, config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
  
)
from sklearn.preprocessing import LabelEncoder

import uvicorn

api = FastAPI()

# model = AutoModelForSequenceClassification.from_pretrained('./saved_model')
# tokenizer = AutoTokenizer.from_pretrained('./saved_model')
# label_encoder = LabelEncoder()


# def get_type(inputword):
#     # Encode the input word/question
#     new_encodings = tokenizer([inputword], truncation=True, padding=True, max_length=512, return_tensors='pt')

#     # Get model predictions
#     with torch.no_grad():
#         outputs = model(**new_encodings)
#         predictions = torch.argmax(outputs.logits, dim=-1)
    
#     # Convert predictions back to labels
#     predicted_labels = label_encoder.inverse_transform(predictions.cpu().numpy())
    
#     # Return the result
#     return {"type": f'{predicted_labels[0]}'}
 


# Initialize FAISS and embeddings
model_path = "sentence-transformers/all-MiniLM-L12-v2"
HF_embeddings = HuggingFaceEmbeddings(model_name=model_path)

# Load the FAISS vector store
faiss_store = FAISS.load_local("tafasir_quran_faiss_vectorstore", embeddings=HF_embeddings, allow_dangerous_deserialization=True)
retriever = faiss_store.as_retriever()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@api.post("/send-message", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_message = request.message
    print(user_message)
    
    # Search using FAISS Retriever
    search_results = retriever.get_relevant_documents(query=user_message,
                                                    #   filters=get_type(user_message),
                                                      top_k=5)
    retrieved_texts = [doc.metadata['answer'] for doc in search_results]
    print(retrieved_texts)
    input_messages = [HumanMessage(content=user_message),]

    # Process the model's response
    try:
        output = app.invoke({"messages": input_messages, "retrieved_texts": retrieved_texts}, config, output_keys=["messages"], stream_mode="values")['messages']
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing response")

    ai_response = ""
    for chunk in output:
        if isinstance(chunk, AIMessage):
            input_messages.append(chunk)
            ai_response += chunk.content
    
    return ChatResponse(response=ai_response)