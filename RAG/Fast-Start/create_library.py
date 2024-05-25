import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.setup import Setup
from llmware.configs import LLMWareConfig

def parsing_documents_into_library(library_name, selected_folder):

    library = Library().create_new_library(library_name)

    sample_files_path = Setup().load_sample_files(over_write=False)

    ingestion_folder_path = os.path.join(sample_files_path, selected_folder)

    parsing_output = library.add_files(ingestion_folder_path)

    updated_library_card = library.get_library_card()
    
    doc_count = updated_library_card['documents']
    block_count = updated_library_card['blocks'] 

    library_path = library.library_main_path

    # Running at test

    test_query = 'executive employment agreement'

    doc_filter = {'doc_id': [3]}
     
    query_results = Query(library).text_query_with_document_filter(test_query, doc_filter=doc_filter, result_count=3)
    
    for idx, result in enumerate(query_results):

        text = result['text']
        file_source = result['file_source']
        page_number = result['page_num']
        doc_id		= result['doc_ID']
        block_id	= result['block_ID']
        matches 	= result['matches']

        print('Query results: ', idx, result)

    return parsing_output

if __name__ == '__main__':

    LLMWareConfig().set_active_db('sqlite')
    LLMWareConfig().set_config(name='debug_mode', value=2)
    
    sample_folders = ['Agreements', 'Invoices', 'UN-Resolutions-500', 'SmallLibrary', 'FinDocs', 'AgreementsLarge']

    library_name = 'library_000'

    selected_folder = sample_folders[0]

    output = parsing_documents_into_library(library_name, selected_folder)

    
    

