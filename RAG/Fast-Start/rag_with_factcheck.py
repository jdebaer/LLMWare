import os
from llmware.prompts import Prompt, HumanInTheLoop
from llmware.setup import Setup
from llmware.configs import LLMWareConfig
from llmware.retrieval import Query
from llmware.library import Library




def rag_processing(library_name, llm_model_name):

    local_path = Setup().load_sample_files()
    agreements_path = os.path.join(local_path,'AgreementsLarge')

    library = Library().create_new_library(library_name)
    library.add_files(agreements_path)

    # Below an example of initial document filtering based on text we know will appear on the first page of each document.

    query = Query(library)

    # results_only=False -> return value is dict with 4 keys: {'query', 'results', 'doc_ID', 'file_source'}
    msa_query_results = query.text_search_by_page('"master services agreement"', page_num=1, results_only=False)

    msa_doc_names = msa_query_results['file_source']
    msa_doc_ids = msa_query_results['doc_ID']

    prompter = Prompt().load_model(llm_model_name)

    print(f'Documents filtered by page: ', msa_doc_ids)

    # Now we're going to loop through these docs and do filtering (this could be text-based or semantic-based) to find the best blocks.
    
    for idx, msa_doc_id in enumerate(msa_doc_ids):

        current_msa_doc_name = msa_doc_names[idx]

        # Remove any path.
        if os.sep in current_msa_doc_name:
            current_msa_doc_name = current_msa_doc_name.split(os.sep)[-1]

        print(f'*******************************************************************************')
        print(f'Now handling document ', msa_doc_id)

        doc_filter = {'doc_ID': [msa_doc_id]}

        # In the doc for the current msa_doc_id, identify the blocks that match 'termination'. Text query, not semantic query.
        block_query_results = query.text_query_with_document_filter('termination', doc_filter)

        augmented_prompt = prompter.add_source_query_results(block_query_results)

        # To see the augmented prompts, uncomment the line below.
        #print(f'Augmented prompt: ', augmented_prompt)

        llm_responses = prompter.prompt_with_source('What is the notice for termination for convenience?')

        # Fact checking.
        stats = prompter.evidence_comparison_stats(llm_responses)
        evidence_sources = prompter.evidence_check_sources(llm_responses)

        for idj, llm_response in enumerate(llm_responses):
        
            print(f'\nLLM response: ', llm_response['llm_response'])
            print(f'\nCompare with evidence: ', stats[idj]['comparison_stats'])
            print(f'\nSources: ', evidence_sources[idj]['source_review'])

       
        # Clear prompter for next run.
        prompter.clear_source_materials()

    print(f'Prompt JSON state saved at: ', os.path.join(LLMWareConfig.get_prompt_path(), prompter.prompt_id))
    prompter.save_state()

    csv_output_location = HumanInTheLoop(prompter).export_current_interaction_to_csv()
    print(f'Prompt interaction saved in CSV: ', csv_output_location)

    return 0

if __name__ == '__main__':

    LLMWareConfig().set_active_db('sqlite')

    # Use dragon for better results.
    llm_model_name = 'llmware/dragon-yi-6b-gguf'
    #llm_model_name = 'llmware/bling-1b-0.1'

    rag_processing('library_001', llm_model_name)
