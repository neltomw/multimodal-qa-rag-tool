import os
from langchain.vectorstores import PGVector

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4-turbo"
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING")
POSTGRES_DRIVER = os.getenv('POSTGRES_DRIVER')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')

# Function to generate the connection string using PGVector (if needed)
def get_pgvector_connection_string():

    return PGVector.connection_string_from_db_params(
        driver=POSTGRES_DRIVER,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )