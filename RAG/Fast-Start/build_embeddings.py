import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.configs import LLMWareConfig

def install_vector_embeddings(library, embedding_model_name):
    
    library_name = library.library_name

    vector_db = LLMWareConfig().get_vector_db()

    # This one line creates the context vector embeddings, one embedding per block, using the specified model.
    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db, batch_size=100)

    ## For prod regulary request updates.
    #update = Status().get_embedding_status(library_name, embedding_model)
    #print(update)
     
    # Test run.
    sample_query = 'incentive compensation'
    
    query_results = Query(library).semantic_query(sample_query, result_count=3)

    for idx, result in enumerate(query_results):

        text = result['text']
        file_source = result['file_source']
        page_number = result['page_num']
        vector_distance = result['distance']

        if len(text) > 125: text = text[0:125] + ' ... '

        print(text)

    embedding_record = library.get_embedding_status()
    print(embedding_record)

    return 0 
    
    




if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    LLMWareConfig().set_active_db('sqlite')
    LLMWareConfig().set_vector_db('faiss')

    # We use the library we created in create_library.py.
    library_name = 'library_000'
    library = Library().load_library(library_name)
    
    #embedding_models = ModelCatalog().list_embedding_models()
    #for idx, model in enumerate(embedding_models):
    #    print(idx, model)

    #embedding_model_name = 'mini-lm-sbert'
    embedding_model_name = 'industry-bert-contracts'

    # embedding model_name = 'text-embedding-ada-002'
    # os.environ['USER_MANAGED_OPENAI_API_KEY'] = '<put API key here>'
    
    install_vector_embeddings(library, embedding_model_name)

