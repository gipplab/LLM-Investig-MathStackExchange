import os
import argparse
import torch
from timer import timer_begin, timer_end, timer_report

model_dict = {
    "tora-7b": "llm-agents/tora-code-7b-v1.0",
    "tora-13b": "llm-agents/tora-code-13b-v1.0",
    "mammoth": "TIGER-Lab/MAmmoTH-Coder-7b",
    "llemma": "EleutherAI/llemma_7b",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}


def psg_encoder__llm(gpu_dev, llm):
    """
    tora-based encoder.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    INSTRUCTIONS = {
        "summarize_instruct": "This passage:'$ E[X] = \\int_{{-\\infty}}^{{\\infty}} xf(x) dx$' means in one word:'Expectation'. This passage:'(x-a)^2 + (y-b)^2=r^2$' means in one word: 'Circle'. This passage:'The distance between the center of an ellipse and either of its two foci. ' means in one word:'Eccentricity'. This passage:'{text}' means in one word:",
        "summarize": "This sentence:'{text}' means in one word:"
    }

    model_path = model_dict[llm]
    model_name = model_path.split("/")[-1].replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(gpu_dev)
    model.eval()

    def encoder(batch_psg, debug=False):
        """
        take the last hidden state to be the embedding
        """
        instruction = INSTRUCTIONS["summarize_instruct"]
        batch_psg = [instruction.format(text=x) for x in batch_psg]
        inputs = tokenizer(batch_psg, truncation=True, max_length=2048,
                           return_tensors="pt", padding=True)
        # inputs = torch.nn.DataParallel(inputs)
        inputs = inputs.to(gpu_dev)
        if debug:
            print(tokenizer.decode(inputs['input_ids'][0]))
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            # Last token pooling pooling
            last_hidden_state = outputs.hidden_states[-1]
            idx_last_non_padding = inputs.attention_mask.bool().sum(1) - 1
            embedding = last_hidden_state[torch.arange(last_hidden_state.shape[0]), -1]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()

    dim = model.config.hidden_size
    return encoder, (tokenizer, model, dim, model_name)


def indexer__docid_vec_hnsw_faiss(outdir, M, efC, efSearch, dim, display_frq):
    os.makedirs(outdir, exist_ok=False)
    """ 
    encode docs and build faiss index from them. hnsw means only approximate search.
    """
    import pickle
    import faiss
    import faiss.contrib.torch_utils
    # M: This parameter controls the maximum number of neighbors
    # for each layer.
    faiss_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    # efConstruction and efSearch: Increasing this value improves
    # the quality of the constructed graph and leads to a higher
    # search accuracy.
    faiss_index.hnsw.efConstruction = efC
    faiss_index.hnsw.efSearch = efSearch

    doclist = []

    def indexer(i, docs, encoder):
        nonlocal doclist
        # docs is of [((docid, *doc_props), doc_content), ...]
        passages = [psg for docid, psg in docs]
        embs = encoder(passages, debug=(i % display_frq == 0))
        print(embs.shape)
        print(embs)
        faiss_index.add(embs)
        doclist += docs
        return docs[-1][0][0]

    def finalize():
        with open(os.path.join(outdir, 'doclist.pkl'), 'wb') as fh:
            pickle.dump(doclist, fh)
        faiss.write_index(faiss_index, os.path.join(outdir, 'index.faiss'))
        print('Done!')

    return indexer, finalize


def corpus_reader__arqmath3_rawxml(xml_file, selected_lines=None):
    """
    returns generator object with all documents that were specified in selected_lines.
    """
    if selected_lines is None:
        selected_lines = []
    from xmlr import xmliter
    from bs4 import BeautifulSoup

    def html2text(html, preserve):
        soup = BeautifulSoup(html, "html.parser")
        for elem in soup.select('span.math-container'):
            elem.replace_with(elem.text)
        return soup.text

    def comment2text(html):
        soup = BeautifulSoup(html, "html.parser")
        return soup.text

    if 'Posts' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            if '@Body' not in attrs:
                body = None
            else:
                body = html2text(attrs['@Body'], False)
            ID = attrs['@Id']
            if not ID in selected_lines:
                continue
            vote = attrs['@Score']
            postType = attrs['@PostTypeId']
            if postType == "1":  # Question
                title = html2text(attrs['@Title'], False)
                tags = attrs['@Tags']
                tags = tags.replace('-', '_')
                if '@AcceptedAnswerId' in attrs:
                    accept = attrs['@AcceptedAnswerId']
                else:
                    accept = None
                # YIELD (docid, doc_props), contents
                yield (ID, 'Q', title, body, vote, tags, accept), None
            else:
                assert postType == "2"  # Answer
                parentID = attrs['@ParentId']
                body = "Encode this document for retrieval: " + body
                # YIELD (docid, doc_props), contents
                yield (ID, 'A', parentID, vote), body

    elif 'Comments' in os.path.basename(xml_file):
        for attrs in xmliter(xml_file, 'row'):
            if '@Text' not in attrs:
                comment = None
            else:
                comment = comment2text(attrs['@Text'])
            ID = attrs['@Id']
            answerID = attrs['@PostId']
            # YIELD (docid, doc_props), contents
            yield (answerID, 'C', ID, comment), None
    else:
        raise NotImplemented


def post_ids(runfile):
    """
    returns list of tuples containing topic id, document id and rank
    """
    import csv
    post_ids = []
    with open(runfile, 'r') as runs:
        reader = csv.reader(runs, delimiter=' ')
        for line in reader:
            post_ids.append((line[0], line[2], line[3]))
    return post_ids


def index(llm, outdir, runfile, corpus, query_limit=9999999, rank_limit=1001, device='cpu'):
    """
    index the retrieved docs from runfile, limited by query_limit (only retrieved docs for first n topics) 
    and rank_limit (only top m ranked retrieved docs per topic). save index to outdir/index
    """
    from tqdm import tqdm
    selected_ids = post_ids(runfile)
    selected_ids = [post_id for query, post_id, rank in selected_ids
                    if int(query[3:]) <= query_limit and int(rank) <= rank_limit]
    # prepare corpus reader
    corpus_reader = corpus_reader__arqmath3_rawxml
    print(corpus_reader)
    # calculate batch size
    gpu_dev = 'cpu' if device == 'cpu' else "cuda"
    batch_sz = 3

    # prepare tokenizer, model and encoder
    encoder, (tokenizer, model, dim, model_name) = psg_encoder__llm(gpu_dev, llm)
    # prepare indexes
    print('embedding dim:', dim)
    display_frq = 1
    idx_outdir = os.path.join(outdir, llm, "index")
    indexer, indexer_finalize = indexer__docid_vec_hnsw_faiss(
        idx_outdir, 256, 16, 64, dim, display_frq
    )

    # go through corpus and index
    n = len(selected_ids)
    if n is None: n = 0
    print('corpus length:', n)
    while True:
        progress = tqdm(corpus_reader__arqmath3_rawxml(corpus, selected_lines=selected_ids), total=n)
        batch = []
        batch_cnt = 0
        for row_idx, doc in enumerate(progress):
            # doc is of ((docid, *doc_props), doc_content)
            if doc[1] is None: continue  # Task1 Question is skipped
            batch.append(doc)
            if len(batch) == batch_sz:
                index_result = indexer(batch_cnt, batch, encoder)
                progress.set_description(f"Indexed doc: {index_result}")
                batch = []
                batch_cnt += 1

        if len(batch) > 0:
            index_result = indexer(batch_cnt, batch, encoder)
            print(f"Final indexed doc: {index_result}")

        if indexer_finalize() in [None, True]:
            break


def searcher__docid_vec_flat_faiss(idx_dir):
    """
    searches through faiss index
    """
    import faiss
    import pickle
    # read index
    index_path = os.path.join(idx_dir, 'index.faiss')
    doclist_path = os.path.join(idx_dir, 'doclist.pkl')
    faiss_index = faiss.read_index(index_path)
    with open(doclist_path, 'rb') as fh:
        doclist = pickle.load(fh)
    assert faiss_index.ntotal == len(doclist)
    dim = faiss_index.d
    print(f'Index: {idx_dir}, dim: {dim}')
    # initialize searcher
    faiss.omp_set_num_threads(1)

    def searcher(query, encoder, topk=1000, debug=False):
        """
        return top k nearest neighbors
        """
        embs = encoder([query], debug=debug)
        scores, ids = faiss_index.search(embs, topk)
        scores, ids = scores.flat, ids.flat
        # results is a list of (internal_ID, score, doc)
        results = [(i, score, doclist[i]) for i, score in zip(ids, scores)]
        return results

    def finalize():
        pass

    return searcher, finalize


def _topic_process__arqmath_2020_task1_origin(xmlfile, limit=99999999):
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

    print(xmlfile)
    for attrs in xmliter(xmlfile, 'Topic'):
        qid = attrs['@number']
        if int(qid[2:]) > limit:
            continue
        title = attrs['Title']
        post_xml = title + '\n' + attrs['Question']
        query = html2text(post_xml)
        query = "Encode this query for retrieval of relevant documents: " + query
        yield qid, query, None


def TREC_output(hits, queryID, append=False, output_file="tmp.run", name="APPROACH0"):
    """
    writes list of hits to file in TREC format.
    """
    if len(hits) == 0: return
    with open(output_file, 'a' if append else 'w') as fh:
        for i, hit in enumerate(hits):
            print("%s\t%s\t%s\t%u\t%f\t%s" % (
                queryID,
                str(hit['_']),
                str(hit['docid']),
                i + 1,
                hit['score'],
                name
            ), file=fh)
            fh.flush()


def search(llm, data, adhoc_query=None, max_print_res=3, device='cpu', limit=999999, topk=1000, verbose=False):
    """
    search index (path: data/index) for relevant documents from first n (limit=n) topics from 
    ./topics-and-qrels/topics.arqmath-2020-task1-origin.xml. topk results are written
    to data/tora_arqmath3.run
    """
    # map device name
    gpu_dev = 'cpu' if device == 'cpu' else 'cuda'
    # prepare tokenizer, model and encoder
    encoder, (tokenizer, model, dim, model_name) = psg_encoder__llm(gpu_dev, llm)
    # prepare searcher
    verbose = True

    idx_dir = os.path.join(data, llm, "index")
    searcher, searcher_finalize = searcher__docid_vec_flat_faiss(idx_dir)

    # output config
    output_format = 'TREC'
    output_id_fields = [0, 0]
    output_dir = "runs"
    if adhoc_query:
        output_filename = 'adhoc.run'
    else:
        output_filename = f'{llm}_arqmath3_rerank.run'
    outdir = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    # get topics
    topics_file = "./topics.arqmath-2022-task1-or-task3-origin.xml"
    topics = _topic_process__arqmath_2020_task1_origin(topics_file, limit) if adhoc_query is None else [
        ('adhoc_query', adhoc_query)
    ]

    # search
    open(outdir, 'w').close()  # clear output file
    for qid, query, _ in topics:
        print(qid, query)
        timer_begin()
        search_results = searcher(query, encoder, topk=topk, debug=verbose)
        timer_end()
        if verbose:
            for j in range(max_print_res):
                internal_id, score, doc = search_results[j]
                print(internal_id, score)
                print(doc, end="\n\n")

        if output_format == 'TREC':
            def locate_field(nested, xpath):
                if isinstance(xpath, int):
                    return nested[xpath]
                elif len(xpath) == 1:
                    return locate_field(nested, xpath[0])
                elif isinstance(xpath, list):
                    return locate_field(nested[xpath[0]], xpath[1:])

            hits = []
            for internal_id, score, doc in search_results:
                # doc is of ((docid, *doc_props), doc_content)
                # blank = locate_field(doc[0], output_id_fields[0])
                # docid = locate_field(doc[0], output_id_fields[1])
                blank = locate_field(doc, output_id_fields[0])
                docid = locate_field(doc, output_id_fields[1])
                hits.append({
                    "_": blank,
                    "docid": docid,
                    "score": score
                })

            TREC_output(hits, qid, append=True,
                        output_file=outdir, name=llm)
        else:
            assert NotImplementedError
        print()

    timer_report(f'search_tora.timer')
    print('Output:', outdir)
    searcher_finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utilities for running Approach0 and evaluateion')

    parser.add_argument('--llm', type=str, required=True,
                        help="LLM to for encoding: Either tora-7b, tora-13b, llemma, mammoth or mistral")
    parser.add_argument('--query_limit', type=int, required=False,
                        help="Number of queries to be processed")
    parser.add_argument('--rank_limit', type=int, required=False,
                        help="Number of docs to be processed per query")
    parser.add_argument('--runfile', type=str, required=False,
                        help="Path to runfile to be processed")
    parser.add_argument('--outdir', type=str, required=False,
                        help="Index documents")
    parser.add_argument('--index', required=False, action='store_true',
                        help="Index documents")
    parser.add_argument('--search', required=False, action='store_true',
                        help="Search index")
    parser.add_argument('--corpus', type=str, required=False,
                        help="Path to corpus file (in xml-format)")
    parser.add_argument('--topk', type=int, required=False,
                        help="Keep at most top-K hits in results")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help="Verbose output (showing query structures and merge times)")
    parser.add_argument('--device', type=str, required=False,
                        help="Device. gpu or cpu")

    args = parser.parse_args()

    if args.index:
        index(args.llm, args.outdir, args.runfile, args.corpus, args.query_limit, args.rank_limit, args.device)
    else:
        search(args.llm, args.outdir, device=args.device, limit=args.query_limit, topk=args.topk)
