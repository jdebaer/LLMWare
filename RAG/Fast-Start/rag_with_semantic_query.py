import os
import re
from llmware.prompts import Prompt, HumanInTheLoop
from llmware.setup import Setup
from llmware.configs import LLMWareConfig
from llmware.retrieval import Query
from llmware.library import Library

def semantic_rag(lib_name, llm_model_name):
  
    # We use the library we created in create_library.pyi and for which we created embeddings in build_embeddings.py.
    library_name = 'library_000'
    library = Library().load_library(library_name)

    question_list = ["What is the executive's base annual salary?"]   

    query = Query(library)

    doc_id_list = query.list_doc_id()
    print(doc_id_list)

    doc_file_name_list = query.list_doc_fn()
    print(doc_file_name_list)

    prompter = Prompt().load_model(llm_model_name)

    for idx, doc_id in enumerate(doc_id_list):

        print('***********************************************************************************************************')
        print(f'Analyzing contract', str(idx+1), doc_id, doc_file_name_list[idx])

        print('LLM response:')

        for question in question_list:

            doc_filter = {'doc_ID': [doc_id]}                                           # Since we're doing this doc by doc.

            # This filters out the blocks that pertain to the topic, from the docs filtered out by the doc filter. This is retrieval only.
            #query_results = query.semantic_query_with_document_filter(question, doc_filter, result_count=10, embedding_distance_threshold=1.0)
            query_results = query.semantic_query_with_document_filter(question, doc_filter, embedding_distance_threshold=1.0)

            for idx, result in enumerate(query_results):

                text = result['text']
                file_source = result['file_source']
                page_number = result['page_num']
                vector_distance = result['distance']
                doc_id = result['doc_ID']

                if len(text) > 125: text = text[0:125] + ' ... '

                print('Top Retrieval: ',idx, doc_id, vector_distance, text)

            #if verbose:
            #    for jdx, query_result in enumerate(query_results):
            #        print(f'Update: querying document - ', question, jdx, doc_filter, query_result)

            # Here we add the results of the retrieval query to our prompt.
            source = prompter.add_source_query_results(query_results)

            # Here we invoke the model.
            responses = prompter.prompt_with_source(question, prompt_name='default_with_context', temperature=0.3)

            for idk, response in enumerate(responses):
                print(f'Update: LLM response - ', question, re.sub('[\n]', ' ', response['llm_response']).strip())

            # Remove the source from the prompt to ready it for the next run.
            prompter.clear_source_materials()

    #   Save jsonl report to jsonl to /prompt_history folder
    print("\nPrompt state saved at: ", os.path.join(LLMWareConfig.get_prompt_path(),prompter.prompt_id))
    prompter.save_state()

    #   Save csv report that includes the model, response, prompt, and evidence for human-in-the-loop review
    csv_output = HumanInTheLoop(prompter).export_current_interaction_to_csv()
    print("\nCSV output saved at:  ", csv_output)

    return 0
    
        

    

    



if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    LLMWareConfig().set_active_db('sqlite')
  
    vector_db = 'faiss'

    lib_name = 'library_001'

    llm_model_name = 'llmware/bling-1b-0.1'

    semantic_rag(lib_name, llm_model_name)
