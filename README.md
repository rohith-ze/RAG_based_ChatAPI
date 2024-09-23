# RAG_based_ChatAPI
This Flask application is designed to build a document upload and querying system using Firebase, Pinecone, and Google's Gemini model. Here's an overview of how the code works:

Libraries Used:
Flask: Web framework to handle HTTP requests.
Pinecone: Vector database for document indexing and querying.
Firebase (Firestore): Used for storing and retrieving document metadata.
PyPDF2: Extracts text from uploaded PDF files.
Google Gemini: Generates natural language responses to questions.
UUID: Generates unique document IDs.
Key Functionalities:
Firebase Initialization:

The code initializes Firebase using credentials, and firestore is used to interact with the Firestore database.
Pinecone Initialization:

A Pinecone instance is created to store document embeddings.
It checks if the Pinecone index named 'rag' exists, and if not, it creates one with a dimension of 1024 (for embeddings) and cosine similarity as the metric.
Google Gemini Configuration:

The code configures Google Gemini to generate responses using the provided API key.
Core Functions:
process_pdf(file):

Extracts text content from an uploaded PDF file using PyPDF2.
index_document(text_content):

Converts the extracted text into embeddings (random values in this case, ideally these would be generated from a more sophisticated model).
It then upserts the document's embeddings into the Pinecone index with a unique document ID.
store_in_firebase(chat_name, doc_id, text_content):

Stores document metadata (chat name, document ID, and text content) in Firebase Firestore for future reference.
upload_document() (Route: /upload):

Handles the document upload process, where a user submits a PDF file and a chat name. It processes the file, indexes the document, and stores its metadata in Firestore.
query_document() (Route: /query):

Accepts a chat name and a question, retrieves the relevant document metadata from Firestore, and queries Pinecone for relevant sections based on the document's embeddings.
query_pinecone(doc_id, question):

Takes the document ID and question, generates embeddings for the question, and queries Pinecone for relevant sections of the document using cosine similarity.
generate_answer(relevant_sections, question):

Combines the relevant document sections, sends the combined text and question to Google Gemini, and generates a response. The response is then returned as the answer.
Flask Routes:
/:

Renders a homepage (querry.html) where users can upload documents or input queries.
/upload:

Handles the upload of PDF documents and stores their content in Pinecone and Firebase.
/query:

Accepts a chat name and a question, retrieves relevant document sections, and generates a response using the Gemini model.
Summary of Workflow:
Users can upload a PDF and associate it with a chat name.
The PDF is processed into text, converted into embeddings, and stored in Pinecone and Firebase.
Users can then query a document using a chat name and a question. The system fetches the relevant document, queries Pinecone for relevant sections, and uses Google Gemini to generate a response to the question based on the document's content.