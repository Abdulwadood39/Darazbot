from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import speak
# Specify the collection name and the directory where the database is persisted
MODEL_NAME = "mistral"

COLLECTION_NAME = "local-rag"

VECTOR_DB_DIR = "./vector_db"


# HYDE = """As an AI language model assistant you are created for Daraz.pk for Customer and 
# staff queries to be addressed, your goal is to assist users by providing multiple perspectives
# on their queries and retrieving relevant documents from our database. Please generate five different
# versions of the user's question, ensuring clarity and diversity. If the user is seeking product information, 
# incorporate synonyms to enhance search accuracy. If clarification is needed, prompt the user for additional information
# rather than making assumptions. Your aim is to address the limitations of distance-based similarity search.
# following are the categories and sub-categories of products  available in daraz.pk 
# add the category / subcategory to the query if it is related to a product

# 'Accessories categories with sub category of eye wear'
# 'Accessories categories with sub category of mens jewelry'
# 'Accessories categories with sub category of men watches'
# 'Electronics categories with sub category of tv accessories'
# 'Home categories with sub category of bedding'
# 'Fitness categories with sub category of nutrition'
# 'Electronics categories with sub category of landline phones'
# 'Home categories with sub category of kitchen'
# 'Accessories categories with sub category of women bags'
# 'Electronics categories with sub category of laptops'
# 'Fitness categories with sub category of outdoor activities'
# 'Electronics categories with sub category of headphones headsets'
# 'Home categories with sub category of furniture'
# 'Electronics categories with sub category of desktops'
# 'Electronics categories with sub category of camera'
# 'Fitness categories with sub category of sports clothing'
# 'Vehicle categories with sub category of motorcycle'
# 'Accessories categories with sub category of womens jewelry'
# 'Accessories categories with sub category of mens accessories'
# 'Home categories with sub category of laundry'
# 'Electronics categories with sub category of computer laptops'
# 'Electronics categories with sub category of led'
# 'Electronics categories with sub category of camera accessories'
# 'Accessories categories with sub category of womens accessories'
# 'Electronics categories with sub category of audio'
# 'Vehicle categories with sub category of motor cars'
# 'Fitness categories with sub category of team sports'
# 'Electronics categories with sub category of portable speakers'
# 'Fitness categories with sub category of sports accessories'
# 'Electronics categories with sub category of mobile tablets accessories'
# 'Electronics categories with sub category of computing storage'
# 'Home categories with sub category of lightning'
# 'Electronics categories with sub category of computing peripherals accessories'
# 'Electronics categories with sub category of home theater speaker'
# 'Electronics categories with sub category of security camera'
# 'Home categories with sub category of home decoration'
# 'Fitness categories with sub category of fitness gadgets'
# 'Electronics categories with sub category of electronic insurance'
# 'Fitness categories with sub category of racket sports'
# 'Accessories categories with sub category of travels'
# 'Electronics categories with sub category of printers'
# 'Electronics categories with sub category of tablets'
# 'Electronics categories with sub category of wearable technology'
# 'Vehicle categories with sub category of automative'
# 'Electronics categories with sub category of featured phones'
# 'Electronics categories with sub category of gaming consoles'
# 'Accessories categories with sub category of women watches'
# 'Electronics categories with sub category of networking', 'Home categories with sub category of bath'
# 'Accessories categories with sub category of men bags', 'Accessories categories with sub category of kids watches'
# 'Electronics categories with sub category of smart watches'
# 'Fitness categories with sub category of exercise fitness'


# Original Question: {question}

# Perspectives:
# 1. How can I help you with {question}?
# 2. What information are you looking for regarding {question}?
# 3. Are you interested in learning more about this product {question}?
# 4. Do you have any specific queries about {question}?
# 5. Can you provide more details about your inquiry regarding {question}?
# """

# PRE_PROMPT = """Please provide your response based solely on the following context. 
#                 If clarification is required, prompt the user for further details 
#                 to ensure a complete understanding of their requirements, including 
#                 product features, characteristics, and budget, also 
#                 make sure the question and context both are related to Darak.pk 
#                 Context: {context}
#                 Question: {question}
                
#                 NOTE:   give concise and short answers and if their are list of products 
#                         choose the best among them don't enlist all of them.
#                         also if the question and context is not related to Darak.pk then don't answer 
#                         it reply with a sorry message and tell out of context for me in a friendly manner.
#                 """

# HYDE = """As an AI language model assistant, I'm here to assist you with queries related to Daraz.pk products and services. If you have any questions about our offerings, feel free to ask! Please note that my expertise lies within Daraz.pk, so I may not be able to assist with queries outside this domain.

# Original Question: {question}

# Perspectives:
# 1. How can I assist you with {question} on Daraz.pk?
# 2. What specific information do you need about {question} on Daraz.pk?
# 3. Interested in learning more about {question} on Daraz.pk?
# 4. Do you have any particular queries about {question} on Daraz.pk?
# 5. Could you provide more details about your inquiry regarding {question} on Daraz.pk?
# """

# PRE_PROMPT = """Please provide your response based solely on the following context related to Daraz.pk. If clarification is required, feel free to ask for further details to ensure I can help you effectively. Remember, I'm here specifically for Daraz.pk-related queries.

# Context: {context}
# Question: {question}

# NOTE: Please keep your answers concise and related to Daraz.pk. If the question and context are not related to Daraz.pk, I'm afraid I won't be able to provide assistance. Apologies for any inconvenience caused.
# """

HYDE = """Welcome! I'm here to assist you with queries related to Daraz.pk products and services. Please feel free to ask any questions you have about our offerings. However, if your question is unrelated to Daraz.pk, I may not be able to provide assistance.

Original Question: {question}

Perspectives:
1. How can I assist you with {question} on Daraz.pk?
2. What specific information do you need about {question} on Daraz.pk?
3. Interested in learning more about {question} on Daraz.pk?
4. Do you have any particular queries about {question} on Daraz.pk?
5. Could you provide more details about your inquiry regarding {question} on Daraz.pk?
"""

PRE_PROMPT = """Please provide your response based solely on the following context related to Daraz.pk. If clarification is required, feel free to ask for further details to ensure I can help you effectively. Remember, I'm here specifically for Daraz.pk-related queries.

Context: {context}
Question: {question}

NOTE: Please keep your answers concise and related to Daraz.pk. If the question and context are not related to Daraz.pk, I'm afraid I won't be able to provide assistance. Apologies for any inconvenience caused.
"""

                

class Darazbot:
    
    """
        Darazbot is an AI-powered customer support assistant for Daraz.pk.
        It uses a language model to understand and respond to user queries,
        and a vector database to retrieve relevant documents or information.

        Attributes
        ----------
        model : ChatOllama
            The language model used for generating responses.
        vector_db : Chroma
            The vector database used for document retrieval.
        retriever : MultiQueryRetriever
            The retriever that combines the vector database and language model.
        prompt : ChatPromptTemplate
            The template used for generating prompts for the language model.

        Methods
        -------
        query(question)
            Processes a user query and returns a response.
        talk(text)
            Converts text to speech.
    """
    
    def __init__(self, model_name : str = MODEL_NAME,
                 collection_name : str = COLLECTION_NAME,
                 persist_directory : str = VECTOR_DB_DIR,
                 HYDE : str = HYDE,
                 PRE_PROMPT : str = PRE_PROMPT
                 ):
        """
            Initializes the Darazbot with the specified model and database settings.

            Parameters
            ----------
            model_name : str, optional
                The name of the language model to use (default is MODEL_NAME).
            collection_name : str, optional
                The name of the collection in the vector database (default is COLLECTION_NAME).
            persist_directory : str, optional
                The directory where the vector database is persisted (default is VECTOR_DB_DIR).
            HYDE : str, optional
                The template for generating alternative queries (default is HYDE).
            PRE_PROMPT : str, optional
                The template for generating prompts for the language model (default is PRE_PROMPT).
        """
        
        # base_url = "http://ollama-container:11434" #for running via docker use ...
        base_url = "http://localhost:11434"  #for running locally use .....
        
        # Load the model Locally from downloaded Ollama models
        self.model = ChatOllama(model=model_name, base_url=base_url)
        # Load the persisted database
        self.vector_db = Chroma(collection_name=collection_name,
                                embedding_function=OllamaEmbeddings(model="nomic-embed-text", base_url=base_url),
                                persist_directory=persist_directory)

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template=HYDE
        )

        self.retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),
            self.model,
            prompt=QUERY_PROMPT
        )

        self.prompt = ChatPromptTemplate.from_template(PRE_PROMPT)

    def query(self, question):
        
        """
            Processes a user query and returns a response.

            This method first checks if the query is related to documents in the VectorDB.
            If related documents are found, it generates a response based on the query and the retrieved documents.

            Parameters
            ----------
            question : str
                The user's query.

            Returns
            -------
            str
                The response generated by the Darazbot.
        """
        
        #check if the query is related to our documents stored in VectorDB
        context = self.vector_db.similarity_search(question, k=2)
        
        if not context:
            return "I'm sorry, I couldn't find relevant information in the database. Could you please provide more details?"
        else:
            chain = (
                {"context": self.retriever,
                 "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )
            response = chain.invoke(question).replace("\n\n", "\n").replace(".\n", ". ").strip()
            print(response)
            # Check if response is empty or out of scope
            if not response:
                return "I'm sorry, I'm not equipped to answer questions outside the scope of Daraz.pk. Could you please ask a Daraz-related question?"
            else:
                return response
            
    def talk(self, text):
        """
            Converts the given text to speech.

            Parameters
            ----------
            text : str
                The text to be converted to speech.
        """
        
        speak.TTS(text=text)
        


if __name__ == "__main__":
    daraz_customer_support = Darazbot()
    while True:
        question = input("What is your query: ")
        if question.lower() in ["quit", "exit"]:
            break
        else:
            response = daraz_customer_support.query(question=question)
            daraz_customer_support.talk(response)
            print()
