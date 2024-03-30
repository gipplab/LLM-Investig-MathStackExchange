## About
This repository contains the resources used for SIGIR'2024 submission "Can LLMs Master Math? Investigating Large Language Models on Math Stack Exchange"

## Quick Start
### Answer generation
Generate answers to the questions in the Arqmath3 competition dataset with the following call:
```sh
mkdir data
python3 code/genArqmathAnswers.py --llm tora-7b
```

(--llm tora-7b can be modified to any of tora-7b, tora-13b, llemma, mammoth or mistral.)

Answers are saved as `./topics-and-qrels/{llm}/topics.arqmath-2022-{llm}-origin-and-generated-answers-0.csv`

To produce runs, we refer to the https://github.com/approach0/pya0/tree/mabowdor repository. You need to perform a few additions: 

```sh
cp -r topics-and-qrels path-to/pya0/
cp -r code/pya0-replace/* path-to/pya0/
```

Obtain runs via 

```sh
cd path-to/pya0/utils/
python3 -m transformer_eval search path-to/pya0/utils/training-and-inference/inference.ini search__tora_7b_generated_single_vec \
--backbone=cocomae --ckpt=220 --use_prebuilt_index=arqmath-task1-dpr-cocomae-220
```

Then evaluate:
```sh
../eval-arqmath3/task1/preprocess.sh cleanup
../eval-arqmath3/task1/preprocess.sh ./runs/arqmath3-cocomae-220-hnsw-top1000.run
../eval-arqmath3/task1/eval.sh
```

### Generating Embeddings

Build an FAISS index of embeddings for the questions in the Arqmath3 competition dataset, we first have to download the complete set of posts. It can be obtained here:

https://drive.google.com/file/d/14SSwTqLZgLVP6iDsAJbmxgb01a8NYyDb/view

Now build the embeddings with the following call:

```sh
python3 code/genArqmathEmbeddings.py --index --llm tora-7b \
--device gpu --query_limit 100 --rank_limit 10 --outdir embeddings_data --corpus Posts.V1.3.xml \ 
--runfile topics-and-qrels/mergerun--0.4W_arqmath3_a0.run--0.2W_arqmath3-SPLADE-nomath-cocomae-2-2-0-top1000.run--0.4W_arqmath3-cocomae-220-top1000.run
```
(--llm tora-7b can be modified to any of tora-7b, tora-13b, llemma, mammoth or mistral. --rank_limit 10 replicates our setting of top 10 retrieved documents reranked.)

Produce a run from said index by calling:

```sh
python3 code/genArqmathEmbeddings.py --seaarch --llm tora-7b \
--topk 10 --device gpu --query_limit 100 --outdir embeddings_data
```

Indices are found in the folders `./embeddings-data/{llm}/index`.

Runfiles are saved in `./runs/{llm}_arqmath3_rerank.run`.

They can be evaluated within the /pya0 module, by 
```sh
cp ./runs/ path-to/pya0/training-and-inference/runs/
../eval-arqmath3/task1/preprocess.sh cleanup
../eval-arqmath3/task1/preprocess.sh ./runs/arqmath3-cocomae-220-hnsw-top1000.run
../eval-arqmath3/task1/eval.sh
```

## Reference

[A. Satpute, N. Giessing, A. Greiner-Petter, M. Schubotz, O. Teschke, A. Aizawa, and B. Gipp, “Can LLMs Master Math? Investigating Large Language Models on Math Stack Exchange,” in Proceedings of 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), Washington, USA, 2024.](https://gipplab.org/wp-content/papercite-data/pdf/satpute2024b.pdf)


