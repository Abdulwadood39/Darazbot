from langchain_core.documents.base import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

# Define the directory where data files are stored
DATA_DIR = "Data"

if __name__ == "__main__":
    # Open and read the 'daraz.txt' file, storing its content as a single string
    with open(os.path.join(DATA_DIR, 'daraz.txt'), 'r') as file:
        raw_text = file.read()
    # Split the raw text into a list of strings, each representing a separate section
    raw_text = raw_text.split("\n\n\n")
    # Filter out any empty strings from the list
    raw_text = [item for item in raw_text if item]

    # Open and read the 'products.txt' file, storing its content as a single string
    with open(os.path.join(DATA_DIR,'products.txt'), 'r') as file:
        raw_products = file.read()
    # Split the raw products into a list of strings, each representing a separate product
    raw_products = raw_products.split("\n")
    # Filter out any empty strings from the list
    raw_products = [item for item in raw_products if item]

    # Initialize an empty list to store products grouped by category
    products = []
    new_list = None

    # Define a function to check if two products belong to the same category
    def check_product_category(info1, info2):
        # Extract the category information from each product description
        info1 = info1.split("It belongs to the")[-1]
        info2 = info2.split("It belongs to the")[-1]
        # Return True if the categories match, False otherwise
        return info1 == info2

    # Iterate over each product, grouping them by category
    for i, product in enumerate(raw_products):
        new_list = [] if new_list == None else new_list
        if len(new_list) and not check_product_category(new_list[-1], product):
            products.append(new_list)
            new_list = None
        else:
            new_list.append(product)

    # Create Document objects for each section of raw text and each group of products
    documents = [Document(page_content=text) for text in raw_text]
    documents.extend([Document(page_content="\n".join(text)) for text in products])

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Initialize a Chroma vector database with the chunked documents, using OllamaEmbeddings
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag",
        persist_directory="./vector_db"
    )
    # Persist the vector database to disk
    vector_db.persist()
