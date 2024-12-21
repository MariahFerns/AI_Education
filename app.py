import time
import os
import sagemaker, boto3, json
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.embeddings import SagemakerEndpointEmbeddings
from typing import Any, Dict, List, Optional
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma, AtlasDB, FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import zipfile
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st

# --- Refactor code into functions ---

def initialize_sagemaker():
    print('Setting role and SageMaker session manually..')
    bucket = 'sagemakerbucketeducation'
    aws_region = 'us-east-1'

    iam = boto3.client('iam', region_name=aws_region)
    sagemaker_client = boto3.client('sagemaker')
    sagemaker_execution_role_name = 'SagemakerRole'

    aws_role = iam.get_role(RoleName=sagemaker_execution_role_name)['Role']['Arn']

    boto3.setup_default_session(region_name=aws_region, profile_name='default')
    sess = sagemaker.Session(sagemaker_client=sagemaker_client, default_bucket=bucket)

    print('Using bucket ', bucket)
    print(aws_region)
    print(aws_role)

    return sess, aws_role

def deploy_model(sess, aws_role):
    _MODEL_CONFIG_ = {
        "huggingface-text2text-flan-t5-large": {
            "instance type": "ml.m5.large",
            "env": {"SAGEMAKER_MODEL_SERVER_WORKERS": "1", "TS_DEFAULT_WORKERS_PER_MODEL": "1"},
        },
        "huggingface-textembedding-all-MiniLM-L6-v2": {
            "instance type": "ml.m5.xlarge",
            "env": {"SAGEMAKER_MODEL_SERVER_WORKERS": "1", "TS_DEFAULT_WORKERS_PER_MODEL": "1"},
        },
    }

    for model_id in _MODEL_CONFIG_:
        endpoint_name = name_from_base(f"ragchatbot-{model_id}")
        inference_instance_type = _MODEL_CONFIG_[model_id]["instance type"]

        deploy_image_uri = image_uris.retrieve(
            region=None,
            framework=None,
            image_scope="inference",
            model_id=model_id,
            instance_type=inference_instance_type,
        )

        model_uri = model_uris.retrieve(
            model_id=model_id, model_scope="inference"
        )

        model_inference = Model(
            image_uri=deploy_image_uri,
            model_data=model_uri,
            role=aws_role,
            predictor_cls=Predictor,
            name=endpoint_name,
            env=_MODEL_CONFIG_[model_id]["env"],
        )

        model_predictor_inference = model_inference.deploy(
            initial_instance_count=1,
            instance_type=inference_instance_type,
            predictor_cls=Predictor,
            endpoint_name=endpoint_name,
        )
        print(f"Model {model_id} has been deployed successfully.\n")
        _MODEL_CONFIG_[model_id]["endpoint_name"] = endpoint_name

    return _MODEL_CONFIG_

def create_embeddings(aws_region, endpoint_name):
    class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
        def embed_documents(self, texts: List[str], chunk_size: int = 5) -> List[List[float]]:
            results = []
            _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size

            for i in range(0, len(texts), _chunk_size):
                response = self._embedding_func(texts[i : i + _chunk_size])
                results.extend(response)
            return results

    class ContentHandler(EmbeddingsContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
            input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            embeddings = response_json["embedding"]
            return embeddings

    content_handler = ContentHandler()

    embeddings = SagemakerEndpointEmbeddingsJumpStart(
        endpoint_name=endpoint_name,
        region_name=aws_region,
        content_handler=content_handler,
    )

    return embeddings

def create_llm(aws_region, endpoint_name):
    class SMLLMContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json["generated_texts"][0]

    llm_content_handler  = SMLLMContentHandler()

    sm_llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=aws_region,
        model_kwargs={"temperature": 0.3, "max_new_tokens":1024},
        content_handler=llm_content_handler,
    )

    return sm_llm

def load_and_index_data(embeddings):
    # Define paths
    data_dir = "ncert_data"
    unzipped_dir = "ncert_data_unzipped"

    # Unzip folders and load PDFs
    def extract_and_load_pdfs():
        if not os.path.exists(unzipped_dir):
            os.makedirs(unzipped_dir)

        for zip_file in os.listdir(data_dir):
            zip_path = os.path.join(data_dir, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_dir)

    # Read PDFs and extract text
    def load_class_data():
        extract_and_load_pdfs()
        all_text = ""
        for root, _, files in os.walk(unzipped_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    reader = PdfReader(pdf_path)
                    for page in reader.pages:
                        all_text += page.extract_text()
        return all_text

    data = load_class_data()

    # Chunk our documents into smaller sizes for better responses
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(data)
    chunks=text_splitter.create_documents(chunks)

    # Use FAISS to create a vector index from our doc chunks and embeddings FM
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("ncert_chunks_index") #save vector DB to file

    return db

def run_streamlit_app(db, embeddings, sm_llm):
    st.title("Q&A Bot 💬")
    query = st.text_input("What would you like to know?")

    if st.button("Search"):
        db = FAISS.load_local("ncert_chunks_index", embeddings, allow_dangerous_deserialization=True)
        search_results = db.similarity_search(query, k=3)

        with st.spinner("Answering..."):
            template = ''' Answer the question as detailed as possible from the provided context, make  sure to provide all the details. If the answer is not available in the documents, don't provide wrong answer.\n\n
                    Context: \n {context} \n
                    Question: \n {question} \n
                    
                    Answer:
                    '''
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

            llm_chain = load_qa_chain(
                llm=sm_llm,
                prompt=PROMPT,
            )

            response = llm_chain.run({"question": query, "input_documents": search_results})

            # Get the response from GPT-4
            st.markdown("### Answer:")
            st.write(response)

# --- Main execution ---

if __name__ == "__main__":
    sess, aws_role = initialize_sagemaker()
    #_MODEL_CONFIG_ = deploy_model(sess, aws_role)  # Only deploy once, then comment out
    
    # Update these endpoint names with your actual deployed endpoint names
    embedding_endpoint_name = 'ragchatbot-huggingface-textembedding-al-2024-12-21-08-26-59-443'  
    llm_endpoint_name = 'ragchatbot-huggingface-text2text-flan-t-2024-12-21-08-16-41-882'
    
    embeddings = create_embeddings(aws_region='us-east-1', endpoint_name=embedding_endpoint_name)
    sm_llm = create_llm(aws_region='us-east-1', endpoint_name=llm_endpoint_name)

    #db = load_and_index_data(embeddings)  # Only index once, then comment out and load from disk
    db = FAISS.load_local("ncert_chunks_index", embeddings, allow_dangerous_deserialization=True)  # Load from disk after initial indexing

    run_streamlit_app(db, embeddings, sm_llm)