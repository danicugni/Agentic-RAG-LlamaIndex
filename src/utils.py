import os
import requests
import zipfile
import nest_asyncio
from dotenv import load_dotenv
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.core import load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import (LLM_MODEL, 
                    TEMPERATURE, 
                    EMBEDDING_MODEL, 
                    CHUNK_SIZE, 
                    DATA_DIRECTORY, 
                    ZIP_PATH, 
                    YEARS)

nest_asyncio.apply()


def set_global_settings(llm_model:str = LLM_MODEL, temperature:float = TEMPERATURE, embedding_model:str = EMBEDDING_MODEL, 
                        chunk_size:int = CHUNK_SIZE):
    
    load_dotenv()

    Settings.llm = AzureOpenAI(
        engine=llm_model, 
        model=llm_model, 
        temperature=temperature
        )

    Settings.embed_model = HuggingFaceEmbedding(model_name = embedding_model)

    Settings.chunk_size = chunk_size


def download_data(data_directory:str = DATA_DIRECTORY, zip_path:str = ZIP_PATH):
    
    os.makedirs(data_directory, exist_ok=True)
    url = 'https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1'
    zip_path = zip_path

    # os.makedirs("data", exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File scaricato con successo: {zip_path}")
    else:
        print(f"Errore nel download. Status code: {response.status_code}")
        exit(1)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_directory)
        print(f"File ZIP estratto con successo nella directory: {data_directory}")
    except zipfile.BadZipFile:
        print("Errore: il file scaricato non è un archivio ZIP valido.")


def ingest_data(data_directory:str = DATA_DIRECTORY, years:list = YEARS):

    loader = UnstructuredReader()
    doc_set = {}
    all_docs = []
    
    for year in years:
        # full_path = Path("./" + data_directory + "/UBER_" + year + ".html")
        full_path = Path(data_directory) / f"UBER_{year}.html"
        year_docs = loader.load_data(file=full_path, split_documents=False)
        for d in year_docs:
            d.metadata = {"year": year}
        doc_set[year] = year_docs
        all_docs.extend(year_docs)
    
    return doc_set,all_docs #maybe all_docs not useful
    

def create_vector_indices(data_directory:str = DATA_DIRECTORY, years:list = YEARS): #TODO storage_path in configuration file
    doc_set, _ = ingest_data(data_directory, years)

    index_set = {}
    for year in years:
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            doc_set[year],
            storage_context=storage_context,
            show_progress=True
        )
        index_set[year] = cur_index
        storage_context.persist(persist_dir=f"./storage/{year}")
    
    return index_set


def load_existing_indices(years:list = YEARS): #TODO storage_path in configuration file
    index_set = {}
    for year in years:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{year}"
        )
        cur_index = load_index_from_storage(
            storage_context,
        )
        index_set[year] = cur_index
    
    return index_set


def get_indices():
    ...


def single_query_engine_tools(index_set:dict, years:list = YEARS):
    individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
        ),
    )
    for year in years
    ]

    return individual_query_engine_tools


def multiple_query_engine_tools(index_set:dict, years:list = YEARS):

    individual_query_engine_tools = single_query_engine_tools(index_set, years)
    query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools
    )

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="sub_question_query_engine",
            description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
            ),
            )
    
    return query_engine_tool