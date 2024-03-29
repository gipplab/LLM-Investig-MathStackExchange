import argparse
import os
import torch
# from vllm  import LLM,SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dict = {
    "tora-7b": "llm-agents/tora-code-7b-v1.0",
    "tora-13b": "llm-agents/tora-code-13b-v1.0",
    "mammoth": "TIGER-Lab/MAmmoTH-Coder-7b",
    "llemma": "EleutherAI/llemma_7b",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"  
}

INSTRUCTIONS = {
    "ir_instruct": {
        "query": 'Prove that there is an infinite number of primes. Read the passage and generate an answer. Assume for contradiction that there is a maximal prime number $m$. Then let $P$ be the product of all prime numbers up to $m$. If $P+1$ is itself not a prime number, it must be divided by one of the non-trivial factors of $P$, $q$ say. But then $q$ must also divide $(P+1)-P=1$, so it must be $1$, which is a contradiction. {text} Read the passage and generate an answer.',
        "key": 'Assume for contradiction that there is a maximal prime number $m$. Then let $P$ be the product of all prime numbers up to $m$. If $P+1$ is itself not a prime number, it must be divided by one of the non-trivial factors of $P$, $q$ say. But then $q$ must also divide $(P+1)-P=1$, so it must be $1$, which is a contradiction. Read the passage and generate the question that the passage is answering. Prove that there is an infinite number of primes. {text} Read the passage and generate the question that the passage is answering.',
    },
   "ir": {
        "query": '{text} Read the passage and generate an answer.'
    },
    "ir-instruct-mistral": {
        "query": '[INST] {text} Read the passage and generate an answer. [\INST]'
    }
}

def _topic_process__arqmath_2020_task1_origin(xmlfile, lower_limit = 0, upper_limit = 99999):
    """
    returns generator object of first n topics, where limit=n
    """
    from xmlr import xmliter
    from bs4 import BeautifulSoup
   
    def html2text(html):
        soup = BeautifulSoup(html, "html.parser")
        for elem in soup.select('span.math-container'):
            elem.replace_with(elem.text)
        return soup.text
    #topics without qrels
    no_qrels_list = [11,21,23,34,35,36,41,43,51,67,74,77,86,90,92,93,95,96,97,98]
    no_qrels_list = [f'A.3{x}' for x in no_qrels_list]
    for attrs in xmliter(xmlfile, 'Topic'):
        qid = attrs['@number']
        if int(qid[3:])<lower_limit or int(qid[3:])>upper_limit or qid in no_qrels_list:
            continue
        title = attrs['Title']
        post_xml = title + '\n' + attrs['Question']
        query = html2text(post_xml)
        yield qid, query, None

def genAnswersBatch(llm,partition=0,step_size=100):
    xmlfile = 'topics.arqmath-2022-task1-or-task3-origin.xml'
    if llm == 'mistral':
        instruction = INSTRUCTIONS["ir-instruct-mistral"]
    else:
        instruction = INSTRUCTIONS["ir"]

    # getting queries
    queries = list(query for qid, query,_ in _topic_process__arqmath_2020_task1_origin(xmlfile,lower_limit=step_size*partition,upper_limit = step_size*partition+step_size))
    queries_only = queries
    qids = list(qid for qid,query,_ in _topic_process__arqmath_2020_task1_origin(xmlfile,lower_limit = step_size*partition, upper_limit = step_size*partition+step_size))
    queries = [instruction["query"].format(text=query) for query in queries]
    batch_size = 1
    queries_batch = [queries[k*batch_size:(k+1)*batch_size] for k in range(len(queries)//batch_size)]
    queries_only_batch = [queries_only[k*batch_size:(k+1)*batch_size] for k in range(len(queries_only)//batch_size)]
    qids_batch = [qids[k*batch_size:(k+1)*batch_size] for k in range(len(qids)//batch_size)]
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    print("available_gpus: ",available_gpus)
    model_path = model_dict[llm]
    model_name = model_path.split("/")[-1].replace("_","-")
    #llm = LLM(model=model_path, tensor_parallel_size=len(available_gpus)) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_path)
    with torch.no_grad():
        for qid_batch, query_batch, query_pure_batch in zip(qids_batch,queries_batch, queries_only_batch):
            #generate Answers
            #query_outputs = llm.generate(query_batch, SamplingParams(temperature=1,top_p=1,max_tokens=1,n=1,stop= ["</s>", "```output"]))
            #query_outputs = sorted(query_outputs, key=lambda x: int(x.request_id))
            query_batch = tokenizer(query_batch, padding=True,truncation=True, return_tensors='pt')
            query_outputs = model.generate(**query_batch, do_sample=True, num_beams=1, max_new_tokens=500) 
            query_outputs = tokenizer.batch_decode(query_outputs)
            #Save Answers. Need to remove input from the generated answers in a next step.
            with open(f"data/topics.arqmath-2022-{model_name}-generated-answers-{partition}.csv","a") as out_file:
                for qid,output in zip(qid_batch,query_outputs):
                    print("qid: ", qid)
                    print(output)
                    out_text = ' '.join(output.split())
                    #out_text = ' '.join(output.outputs[0].text.split())
                    out_file.write('{num}\t{text}\n'.format(num=qid,text=out_text))
            with open(f"topics-and-qrels/topics.arqmath-2022-{model_name}-origin-and-generated-answers-{partition}.csv","a") as out_file:
                for qid,output,query in zip(qid_batch,query_outputs,query_pure_batch):
                        #out_no_linebreak = ' '.join(output.outputs[0].text.split())
                        out_no_linebreak = ' '.join(output.split())
                        query_no_linebreak = ' '.join(query.split())
                        out_file.write('{num}\t{q}\t{text}\n'.format(num=qid,text=out_no_linebreak,q=query_no_linebreak))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utilities for running Approach0 and evaluation')
    parser.add_argument('--llm', type=str, required=True,
        help="LLM for generation: Either tora-7b, tora-13b, llemma, mammoth or mistral")
    parser.add_argument('partition', type=int)
    parser.add_argument('step_size',type=int)
    args = parser.parse_args()    
    genAnswersBatch(args.llm, args.partition,args.step_size)
