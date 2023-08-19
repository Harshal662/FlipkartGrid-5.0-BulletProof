import os
import fitz  # PyMuPDF
from transformers import GPT2TokenizerFast
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from flask import Flask, render_template, request
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"

# Load the PDF file
pdf_path = "./new.pdf"
try:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    chunks = pages
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
except Exception as e:
    print(f"Error loading PDF: {e}")
    text = ""

# Initialize tokenizer, splitter, embeddings, and models
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

with open('attention_is_all_you_need.txt', 'w', encoding="utf-8") as f:
    f.write(text)

# Define the text splitting function and embedding method (you may need to modify this)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# Create the conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0.1), db.as_retriever())

chat_history = []

# Define routes and views


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    user_query = request.form['user_input']
    response = ""

    if user_query.lower() == 'exit':
        response = "Thank you for using the Fashion Recommender chatbot!"
        chat_history.clear()
    else:
        try:
            # Accumulate the conversation history
            chat_history.append((user_query, ""))

            # Create 'context' by joining the chat history
            context = "\n".join(
                [f'User: {item[0]}\nChatbot: {item[1]}' for item in chat_history])

            # Provide the expected format with 'question' and 'context'
            query_dict = {
                "question": user_query,
                "chat_history": chat_history  # Provide the chat history
            }

            result = qa(query_dict)
            # Update the last entry with the chatbot's response
            chat_history[-1] = (user_query, result['answer'])
            response = result['answer']
        except Exception as e:
            response = "An error occurred but the chatbot will continue running: " + \
                str(e)

    # Render the HTML template with the updated chat history
    return render_template('index.html', chat_history=chat_history, response=response)


if __name__ == '__main__':
    app.run()
