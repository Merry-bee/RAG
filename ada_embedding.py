import openai
import argparse
import torch
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text1", type=str, help="eg: data/10.1086")
    parser.add_argument("--text2", type=str, help="eg: data/10.1086")
    args = parser.parse_args()
    return args

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def main():
    openai.api_key = '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'
    while True:
        text1 = input('text1:')
        text2 = input('text2:')
        # args = get_args()
        emb1 = get_embedding(text1, model='text-embedding-ada-002')
        emb2 = get_embedding(text2, model='text-embedding-ada-002')
        vec1 = torch.FloatTensor(emb1)
        vec2 = torch.FloatTensor(emb2)

        cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
        print('cos_sim:',cos_sim)

if __name__ == '__main__':
    main()