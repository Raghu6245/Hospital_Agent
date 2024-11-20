import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
#Path to the csv file
REVIEWS_CSV_PATH = "data/reviews.csv"
#Path where we store the vector database
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()
#Creating loader using csv path and "review" column in csv file
loader = CSVLoader(file_path= REVIEWS_CSV_PATH, source_column ="review")
reviews = loader.load()
# Creating chroma vector DB
reviews_vector_db = Chroma.from_documents(reviews, OpenAIEmbeddings(), persist_directory = REVIEWS_CHROMA_PATH)
# reviews_vector_db = Chroma(persist_directory=REVIEWS_CHROMA_PATH, embedding_function=OpenAIEmbeddings(),)