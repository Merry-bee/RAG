import torch
import torch.nn.functional as F
import openai
from ada_embedding import get_embedding
import pickle


def recall_topk(new_hypothesis,k,embedding_file):
    emb1 = get_embedding(new_hypothesis, model='text-embedding-ada-002')
    vec1 = torch.FloatTensor(emb1)
    with open(embedding_file,'rb')as fi:
        emb2text = pickle.load(fi)
    embeddings = [torch.FloatTensor(emb) for emb in emb2text.keys()]
    emb2sim = {emb: F.cosine_similarity(vec1, emb, dim=0) for emb in embeddings}
    emb2sim = dict(sorted(emb2sim.items(),key=lambda x: x[1], reverse=True)[:k])
    text2sim = {emb2text[tuple(k.tolist())][1]: emb2sim[k] for k in emb2sim.keys()}
    return text2sim

def main():
    openai.api_key = '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'
    new_hypothesis = input('new hypothesis:')
    text2sim = recall_topk(new_hypothesis,k=5,embedding_file='data/embeddings.pkl')
    print(text2sim)

if __name__ == '__main__':
    main()