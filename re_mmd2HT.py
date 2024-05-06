import re
from pathlib import Path
import json
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmd-path", type=str, help="eg: data/10.1086/correct")
    args = parser.parse_args()
    return args

def mmd2json_HT(mmd_path,mmd_out_path,file_name):
    '''
    找到与H-T相关的章节
    正则匹配
    '''
    file_path = Path(mmd_path) / Path(file_name)
    with open(file_path,'r',errors='ignore')as fi:
        lines = fi.readlines()
    lines = ''.join(lines)
    lines = lines.replace('\n\n','\n')

    dct_HT = {'hypotheses':[],'theories':[]}
    hypotheses = re.findall(r'((\*)+H\d.*?\n)|(H\d:.*?\n)',lines)     # list of tuples: (**H1** Balabala )|(H1: Balabala)
    for h in hypotheses:
        dct_HT['hypotheses'].append(h[0])
    sentences = re.split(r'\.|\n', lines.replace('et al.','et al'))
    theories = [s for s in sentences if re.search(r'\([A-Z].*? [0-9]{4}\)', s)] # . Some balabala (Craik et al. 1996)
    for t in theories:
        t = t[:t.find(')')+1]
        dct_HT['theories'].append(t+'\n')
    with open(mmd_out_path/file_name.with_suffix('.json'),'w') as fo:
        json.dump(dct_HT,fo)
def mmd2txt_keyword(mmd_path,mmd_out_path,file_name,keywords,re_keywords):
    '''
    找到与H-T相关的章节
    正则匹配
    '''
    file_path = Path(mmd_path) / Path(file_name)
    theory_path = mmd_out_path/file_name.with_suffix('.txt')
    if os.path.exists(theory_path):
        return

    with open(file_path,'r',errors='ignore')as fi:
        lines = fi.read()

    paragraphs = lines.replace('\n\n','\n').split('\n')

    theory_paras = []    # 含有theory的段落
    for paragraph in paragraphs:
        if 'References' in paragraph:
            break
        for keyword in keywords:
            if keyword in paragraph:
                theory_paras.append(paragraph.strip()+'\n')
                break
        for re_keyword in re_keywords:
            if re.search(re_keyword,paragraph):
                theory_paras.append(paragraph.strip()+'\n')
                break

    if len(theory_paras) > 0:
        with open(theory_path,'w') as fo:
            fo.writelines(theory_paras)

def mmd2txt_Abstract(mmd_path,mmd_out_path,file_name):
    '''
    找到Abstract段落
    正则匹配
    '''
    file_path = Path(mmd_path) / Path(file_name)
    txt_out_path = mmd_out_path/file_name.with_suffix('.txt')
    
    with open(file_path,'r',errors='ignore')as fi:
        lines = fi.read()

    abstract = re.search(r'# Abstract\n\n.*?\n',lines,re.S)
    if abstract:
        with open(txt_out_path,'w') as fo:
            fo.write(abstract.group(0))


if __name__ == '__main__':
    args = get_args()
    mmd_path = args.mmd_path
    # mmd_out_path = Path(mmd_path).parent / Path('theories')
    mmd_out_path = Path(mmd_path).parent / Path('demonstration')
    mmd_out_path = Path(mmd_path).parent / Path('abstract')
    os.makedirs(mmd_out_path,exist_ok=True)
    for file_name in os.listdir(mmd_path):
        # mmd2txt_keyword(mmd_path,mmd_out_path,Path(file_name),keywords=['theory','effect','model','strategy'],re_keywords=[])
        # mmd2txt_keyword(mmd_path,mmd_out_path,Path(file_name),keywords=[],re_keywords=[r'moderat',r'mediat',r'\((\w*?(,| )){0,1}\d{4}\)'])
        mmd2txt_Abstract(mmd_path,mmd_out_path,Path(file_name))
        

