from flask import Flask, request, jsonify, render_template
import pinecone
import firebase_admin
from firebase_admin import credentials, firestore
import PyPDF2
import uuid
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as gai 
import random
import os
import re

app=Flask(__name__)

#Firebase credential initializing
firebase=credentials.Certificate('C:\\firebase_credential\\tsting-e3e43-firebase-adminsdk-2w2wy-fbb931d857.json')
firebase_admin.initialize_app(firebase)
db=firestore.client()

#Pinecone
pc=Pinecone(api_key="API from Pinecone")
index_name='rag'

if index_name not in pc.list_indexes().names():
    pc.create_index(name='rag', dimension=1024, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))

#Connect to Pinecone index
pinecone_index=pc.Index(index_name)

#Google Gemini config
gai.configure(api_key="Google Gemini API")

#process PDF and extract text
def process_pdf(file):
    reader=PyPDF2.PdfReader(file)
    text=''
    for page in reader.pages:
        text+=page.extract_text()
    return text

#index document in Pinecone
def index_document(text_content):
    embeddings=generate_embeddings(text_content)
    doc_id=str(uuid.uuid4())
    pinecone_index.upsert([(doc_id, embeddings)])
    return doc_id

#generate embeddings for doc
def generate_embeddings(text):
    return[random.uniform(-1.0, 1.0) for _ in range(1024)]

#store document details in Firebase Firestore
def store_in_firebase(chat_name, doc_id, text_content):
    data={
        'chat_name':chat_name,
        'document_id':doc_id,
        'text_content':text_content
    }
    db.collection('documents').add(data)

#GuardRail
def validate_question(question):
    #if the question is too short
    if len(question) < 5:
        return False, "The question is too short. Please provide more detail."

    #Check non-sensical input
    if re.match(r'^[\W_]+$', question):
        return False, "The question seems to be non-sensical or contains only symbols. Please rephrase."

    #Check for offensive lang
    offensive_words=['stupid','bloody']  
    if any(offensive_word in question.lower() for offensive_word in offensive_words):
        return False, "The question contains inappropriate language. Please rephrase."

    return True, ""

#user uploads PDF
@app.route('/')
def index():
    return render_template('rag.html')

#uploading and processing PDF
@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error':'No file part'}), 400
    file=request.files['file']
    chat_name=request.form.get('chat_name')

    if file.filename == '' or not chat_name:
        return jsonify({'error':'Missing file or chat name'}), 400

    text_content=process_pdf(file)
    doc_id=index_document(text_content)
    store_in_firebase(chat_name, doc_id, text_content)

    return jsonify({'message':'Document uploaded and indexed successfully.'}), 200

#query from document
@app.route('/query', methods=['POST'])
def query_document():
    data=request.json
    chat_name=data.get('chat_name')
    question=data.get('question')

    if not chat_name or not question:
        return jsonify({'error':'Missing chat_name or question'}), 400

    #Validate the question
    is_valid, message=validate_question(question)
    if not is_valid:
        return jsonify({'error':message}), 400

    #Query Firebase to get pdf by chat name
    document_ref=db.collection('documents').where('chat_name', '==', chat_name).get()

    if not document_ref:
        return jsonify({'error':'Document not found for the given chat_name'}), 404

    document=document_ref[0].to_dict()
    doc_id=document['document_id']

    relevant_sections=query_pinecone(doc_id, question)

    if not relevant_sections:
        return jsonify({'error':'No relevant sections found for the given question'}), 404

    response=generate_answer(relevant_sections, question)

    return jsonify({'answer':response}), 200

#query pinecone for info based on the question
def query_pinecone(doc_id, question):
    question_embeddings=generate_embeddings(question)

    if len(question_embeddings) != 1024:
        return {"error":"Invalid embedding length"}, 400

    try:
        response=pinecone_index.query(queries=[question_embeddings], top_k=5, include_metadata=True)
    except Exception as e:
        return {"error":str(e)}, 400

    relevant_sections=[]
    for match in response['matches']:
        relevant_sections.append(match['metadata']['text'])

    return relevant_sections

#generate an answer using Google Gemini
def generate_answer(relevant_sections, question):
    combined_text=" ".join([section if isinstance(section, str) else str(section) for section in relevant_sections])

    model=gai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content(f"Here is some document context:{combined_text}. Based on this, answer the following question:{question}")

    if hasattr(response, 'candidates') and len(response.candidates) > 0:
        answer=response.candidates[0].content.parts[0].text
    else:
        answer="Could not generate an answer."

    return answer

#Run Flask 
if __name__ == '__main__':
    app.run(debug=True)
