import pickle
import argparse
import openai
from pathlib import Path
from ada_embedding import get_embedding
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--HT_path", type=str, help="eg: data/10.1086/GPT_HT")
    parser.add_argument("--save_file", type=str, help="eg: data/embeddings.pkl")
    args = parser.parse_args()
    return args

def save_emb(HT_path,save_file):
    start_idx=0
    for file_idx,file in enumerate(os.listdir(HT_path)[start_idx:]):
        with open(save_file, 'rb') as fo:
            dct = pickle.load(fo)

        lines = [v[1] for v in dct.values()]
        with open(os.path.join(HT_path,file),'r') as fi:
            file_lines = fi.readlines()
        for line in file_lines:
            if line and line not in lines:
                try:
                    emb = get_embedding(line)
                    dct[tuple(emb)] = (Path(HT_path).joinpath(file),line)
                except Exception as e:
                    print(f'{e}: {file_idx+start_idx}. {file}')
                    continue
        with open(save_file, 'wb') as fo:
            pickle.dump(dct, fo)
        print(f'{file_idx+start_idx}: {file} done.')






def main():
    openai.api_key = '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'
    args = get_args()
    HT_path = args.HT_path
    save_file = args.save_file
    save_emb(HT_path,save_file)

if __name__ =='__main__':
    main()
