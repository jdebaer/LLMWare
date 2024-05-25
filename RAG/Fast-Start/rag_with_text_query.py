import os
import re
from llmware.prompts import Prompt, HumanInTheLoop
from llmware.setup import Setup
from llmware.configs import LLMWareConfig
from llmware.retrieval import Query
from llmware.library import Library

# Here we retrieve blocks from the DB, but based on a text-based 'topic' that we use to filter on. So we don't do semantic search.

def contract_analysis(model_name, verbose=False):

    # We use the library we created in create_library.py.
    library_name = 'library_000'
    library = Library().load_library(library_name)

    # Topic is to enhance retrieval, not generation.
    question_list = [	{'topic':'executive employment agreement',	'llm_query':'What are the names of the two parties?'},
			{'topic':'base salary',				'llm_query':"What is the executive's base salary?"},
			{'topic':'governing law',			'llm_query':'What is the governing law?'}]

    query = Query(library)

    doc_id_list = query.list_doc_id()
    print(doc_id_list)

    doc_file_name_list = query.list_doc_fn()
    print(doc_file_name_list)

    prompter = Prompt().load_model(model_name)

    for idx, doc_id in enumerate(doc_id_list):
   
        print(f'Analyzing contract', str(idx+1), doc_id, doc_file_name_list[idx])

        print('LLM response:')

        for question in question_list:

            topic = question['topic']
            llm_query = question['llm_query']
            doc_filter = {'doc_ID': [doc_id]}						# Since we're doing this doc by doc.
            
            # This filters out the blocks that pertain to the topic, from the docs filtered out by the doc filter. This is retrieval only.
            query_results = query.text_query_with_document_filter(topic, doc_filter, result_count=3, exact_mode=True)

            if verbose:
                for jdx, query_result in enumerate(query_results):
                    print(f'Update: querying document - ', topic, jdx, doc_filter, query_result)

            # Here we add the results of the retrieval query to our prompt.
            source = prompter.add_source_query_results(query_results)

            # Here we invoke the model.
            responses = prompter.prompt_with_source(llm_query, prompt_name='default_with_context', temperature=0.3)

            for idk, response in enumerate(responses):
                print(f'Update: LLM response - ', llm_query, re.sub('[\n]', ' ', response['llm_response']).strip())

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

    LLMWareConfig().set_active_db('sqlite')

    model_name = 'llmware/bling-1b-0.1'

    contract_analysis(model_name)
