# load llama index libraries

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter


# api key

api_key = "your API Key"  #store in env file 

# Load data

reader = SimpleDirectoryReader("./data")
documents = reader.load_data()


# Define LLM and embedding model
llm = NVIDIA(api_key=api_key,model="meta/llama-3.1-8b-instruct")
embed_model = NVIDIAEmbedding(api_key=api_key, model_name="nvidia/nv-embedqa-e5-v5")
Settings.llm = llm
Settings.embed_model = embed_model


# split the document into small chunks and store it in vectorstore 
node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes, embed_model= embed_model)




# Create query engine
query_engine = index.as_query_engine(llm =llm, similarity_top_k=5)

"""
# your question will be converted using the same embedding model 
and then top chunks close to this are then sent to LLM which will 
answer for you

"""
response = query_engine.query("What are some characteristics of lions? Dont add new information. Only answer from the text file"
)
print(response)



