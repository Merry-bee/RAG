import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time
import os
import json
import argparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor,wait


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="eg: data/10.1086")
    args = parser.parse_args()
    return args

import tiktoken
def count_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chat_HT_json(mmd_out_path,result_path):
    '''
    prompt TODO:
    加入没有显式 H1的样例；
    加入没有作者和年份的样例；
    '''
    exist_output = os.listdir(mmd_out_path)
    start = 519
    for i,json_file in enumerate(os.listdir(mmd_out_path)[start:]):
        if Path(json_file).with_suffix('.txt') in exist_output:
            continue
        prompt_conv = [
            {"role": "system",
             "content": "You are an academic assistant. You will find out the hypotheses and their supporting theories with citation in the given contents."},
            {"role": "user", "content":
                '''
                Hypotheses in this paper:
                **H1:**Relative to taking photos for the self, taking photos with the intention to share with others will reduce enjoyment of an experience. 
                **H2:**Relative to taking photos for the self, taking photos with the intention to share with others will increase self-presentational concern. 
                **H3:**Self-presentational concern will diminish enjoyment both directly and indirectly through reduced engagement in the experience.
                Please select from the theories below to support each hypothesis:
                Further, over 70 million photos are posted each day on the photo-sharing platform Instagram, which recently hit 700 million active users, outpacing micro-blogging site Twitter in both size and growth of its user base (Instagram Press, 2017; Kharpal, 2015). 
                Indeed, people spend significant time and money engaging in experiences, both ordinary and extraordinary (Bhattacharjee and Mogilner, 2014). 
                One reason experiences are so central to well-being is that they are often shared with others, thus contributing to the value and happiness humans derive from their social relationships (Leary and Baumeister, 2000; Myers, 2000). 
                Telling others about an experience after it has ended can boost people's positive affect and sense of meaning (Lambert et al, 2013; Langston, 1994). 
                Some recent work has shown that the act of photo taking can alter evaluations and memories of an experience compared to not taking photos at all (Barasch et al, 2017; Diehl, Zauberman, and Barasch, 2016).
                In general, people are motivated to present themselves to others in a favorable light (Goffman, 1959).
                Social interactions inherently involve the prospect of being evaluated or judged by others in ways that can influence future outcomes (Leary and Kowalski, 1990; Schlenker and Leary, 1982).
                As a result, social situations often increase people's concerns with self-presentation, or their desire to control the way they appear to real or imagined audiences (Schlenker, 1980; Tedeschi, 1981).
                Indeed, any type of photo can convey information about an individual that others might evaluate, thus activating the self-presentational motive of communicating desired identities to others (Gollwitzer, 1986; Leary and Kowalski, 1990)
                Consistent with this notion, people spend substantial time and effort curating their presence on social media, and are frequently worried about managing their impressions in these contexts (Gonzales and Hancock, 2011; Manago et al, 2008).
                Indeed, concern about conveying a favorable self-image to others often conflicts with maximizing one's own satisfaction (Ariely and Levav, 2000; Mackie and Goethals, 1987).
                Moreover, self-presentational concern is often associated with pressure to make a good impression and self-conscious emotions such as anxiety (Leary, 2007; Miller, 1992).
                Thus, these negative self-conscious emotions and heightened self-awareness may _directly_ reduce hedonic enjoyment (Diener, 1979)", "Second, self-presentational concern might also reduce enjoyment _indirectly_, by decreasing engagement or involvement with an experience (Csikszentmihalyi, 1997; Higgins, 2006).
                Self-presentational concern that arises from considering how others will evaluate one's photos may decrease pleasurable immersion in the experience itself, thus reducing enjoyment (Csikszentmihalyi, 1997; Killingsworth and Gilbert, 2010).
                Because self-presentational concern often triggers the self-conscious emotion of anxiety (Leary 2007; Miller 1992).
                The Trait Self-Consciousness Scale (Fenigstein, Scheier, and Buss 1975; Scheier and Carver 1985)", "To decompose this interaction, we examined the relationship between scores on the trait self-consciousness scale and enjoyment for each photo-taking goal condition (Aiken and West, 1991; Spiller et al, 2013).
                Social interactions with certain audiences heighten the prospect of interpersonal evaluation (Gynther, 1957; Schlenker and Leary, 1982).
                Thus, sharing photos with close others may not induce the same level of self-presentational concern relative to sharing photos with broad audiences on social media or with acquaintances, because people are less likely to expect that close friends would judge them (or change their opinions of them)", ", Houghton et al, 2013).
                Since audience size can affect the extent to which individuals share self-presentational content or feel anxiety (Barasch and Berger, 2014).
                Next, participants responded to two questions about their level of engagement in the bus tour experience: How much did you feel immersed in the bus tour experience? on a seven-point Likert scale from ", "While previous work on sharing experiences has looked separately at actual sharing with either close (Lambert et al, 2013)", "One might argue that taking photos to share involves extrinsic rewards, while taking photos for the self involves only intrinsic rewards from the experience itself, thus accounting for the observed effects (Deci, 1971).
                Prior work on the effect of extrinsic rewards on intrinsic motivation (Deci, 1971; Lepper, Greene, and Nisbett, 1973).
                Moreover, while extrinsic rewards have been shown to reduce persistence and effort, two meta-analyses on this vast area of research have not found significant detrimental effects on self-reported enjoyment (Cameron and Pierce, 1994; Deci, Koester, and Ryan, 1999).
                People share hundreds of millions of photos every day (Facebook, 2017; Systrom, 2014).
                Many restaurants and hotels incorporate hashtags throughout their experiences to encourage consumers to take photos for sharing on social media (Mancini, 2014; Veix, 2013).
                Though companies invest substantial resources to create experiences that maximize consumer enjoyment (Pine and Gilmore, 1999; Schmitt, 1999).
                While unrelated interruptions have been shown to reduce adaptation and increase task enjoyment (Nelson, Meyvis, and Galak, 2009)
                '''
             },
            {"role": "assistant", "content":
                '''
                Hypothesis1: Relative to taking photos for the self, taking photos with the intention to share with others will reduce enjoyment of an experience.
                Supporting theories for Hypothesis1: 
                Concern about conveying a favorable self-image to others often conflicts with maximizing one's own satisfaction (Ariely and Levav, 2000; Mackie and Goethals, 1987). 
                Self-presentational concern is often associated with pressure to make a good impression and self-conscious emotions such as anxiety (Leary, 2007; Miller, 1992). 
                These negative self-conscious emotions and heightened self-awareness may _directly_ reduce hedonic enjoyment (Diener, 1979).

                Hypothesis2: Relative to taking photos for the self, taking photos with the intention to share with others will increase self-presentational concern.
                Supporting theories for Hypothesis2:
                Social interactions inherently involve the prospect of being evaluated or judged by others in ways that can influence future outcomes (Leary and Kowalski, 1990; Schlenker and Leary, 1982). 
                Social situations often increase people's desire to control the way they appear to real or imagined audiences (Schlenker, 1980; Tedeschi, 1981).
                Indeed, any type of photo can convey information about an individual that others might evaluate, thus activating the self-presentational motive of communicating desired identities to others (Gollwitzer, 1986; Leary and Kowalski, 1990)
                people spend substantial time and effort curating their presence on social media, and are frequently worried about managing their impressions in these contexts (Gonzales and Hancock, 2011; Manago et al., 2008). 

                Hypothesis3: Self-presentational concern will diminish enjoyment both directly and indirectly through reduced engagement in the experience.
                Supporting theories for Hypothesis3:
                Self-presentational concern might also reduce enjoyment _indirectly_, by decreasing engagement or involvement with an experience (Csikszentmihalyi, 1997; Higgins, 2006). 
                Self-presentational concern that arises from considering how others will evaluate one's photos may decrease pleasurable immersion in the experience itself, thus reducing enjoyment (Csikszentmihalyi, 1997; Killingsworth and Gilbert, 2010).
                '''
             },

        ]

        # TODO: 如果没有输入完，分几次输入
        '''num_tokens = count_tokens(raw_paper, "cl100k_base")
        if num_tokens>4097:
            pass
        '''

        try:
            with open(mmd_out_path/json_file,'r') as fi:
                dct_HT = json.load(fi)
            if len(dct_HT['hypotheses']) == 0:
                continue  # 没有显示H：先不处理
            content='Hypotheses in this paper:'
            for h in dct_HT['hypotheses']:
                content += h
            content += 'Please select from the theories below to support each hypothesis:'
            for t in dct_HT['theories']:
                content += t
            prompt_conv.append({'role':'user','content':content})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt_conv
            )
            result = response['choices'][0]['message']['content']
            result_file = Path(result_path) / Path(json_file).with_suffix('.txt')
            with open(result_file,'w') as fo:
                fo.write(result)
            # print('success:',json_file)
        except Exception as e:
            print(i+start,json_file,e)
            # break


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def process_theory(input):
    txt_out_path, txt_file, result_path, prompt_conv = input
    with open(txt_out_path/txt_file,'r') as fi:
        lines = fi.readlines()
    result_file = Path(result_path) / Path(txt_file).with_suffix('.json')
    filter_result = []
    break_flag = False
    for start_line_idx in range(0,len(lines),2):
        process_lines = lines[start_line_idx:start_line_idx+2]
        try:
            prompt_conv = prompt_conv[:11]
            process_lines = ''.join([f'{start_line_idx+i+1}. {process_lines[i]}' for i in range(len(process_lines))])
            prompt_conv.append({'role': 'user', 'content': process_lines})

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt_conv
            )
            result = response['choices'][0]['message']['content']

            for line in result.split('\n'):
                if len(line.split(':')) < 2:
                    continue
                theory_name = re.sub(r'\d+\.','',line.split(':')[0]).strip()
                para = line.split(':')[1].strip()
                if theory_name in para and theory_name[-6:] == 'theory':
                    filter_result.append(json.dumps({'theory':theory_name,'paragraph':para})+'\n')

        except Exception as e:
            print(txt_file,e)
            time.sleep(3)
            break_flag = True
            break
    if not break_flag:
        with open(result_file, 'w') as fo:
            fo.writelines(filter_result)
    time.sleep(3)

def process_abstract2idea(input):
    txt_out_path, txt_file, result_path, prompt_conv = input
    with open(txt_out_path/txt_file,'r') as fi:
        abstract = fi.read()
    result_file = Path(result_path) / Path(txt_file)

   
    try:
        prompt_conv = prompt_conv[:5]
        prompt_conv.append({'role': 'user', 'content': abstract})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt_conv
        )
        idea = response['choices'][0]['message']['content']
        with open(result_file, 'w') as fo:
            fo.write(idea)
        

    except Exception as e:
        print(txt_file,e)
        time.sleep(3)

    
        
    time.sleep(3)
    
def chat_theory_txt(txt_out_path,result_path):

    prompt_conv = [
        {"role": "system",
         "content": ""
         },
        {"role": "user", "content": ""
         },

    ]
    # authority information
    with open('data/prompt/txt2theory/system_prompt.txt', 'r') as fi:
        prompt_conv[0]["content"] = fi.read()
    # example and task explaination
    with open('data/prompt/txt2theory/theories_example.txt', 'r') as fi:
        prompt_conv[1]["content"] += fi.read()
    prompt_conv.append({"role": "assistant",
                        "content": "Sure, I understand. Please give me your paragraphs and I will find out the theories and according paragraphs."}, )
    # wrong example and correction
    with open('data/prompt/txt2theory/208501_example.txt', 'r') as fi:
        prompt_conv.append({'role': 'user', 'content': fi.read()})
    with open('data/prompt/txt2theory/208501_wrong_result.txt', 'r') as fi:
        prompt_conv.append({'role': 'assistant', 'content': fi.read()})
    with open('data/prompt/txt2theory/208501_true_result.txt', 'r') as fi:
        prompt_conv.append({'role': 'user', 'content': fi.read()})
    prompt_conv.append({"role": "assistant",
                        "content": "Sorry, I misunderstood your task. I will give you the exact theory names mentioned in the paragraphs."}, )
    # wrong example and correction
    with open('data/prompt/txt2theory/208502_example.txt', 'r') as fi:
        prompt_conv.append({'role': 'user', 'content': fi.read()})
    with open('data/prompt/txt2theory/208502_wrong_result.txt', 'r') as fi:
        prompt_conv.append({'role': 'assistant', 'content': fi.read()})
    with open('data/prompt/txt2theory/208502_true_result.txt', 'r') as fi:
        prompt_conv.append({'role': 'user', 'content': fi.read()})
    prompt_conv.append({"role": "assistant",
                        "content": "Sorry, I misunderstood your meaning. I will give you the same number of paragraphs as your input and just find out the theory name and original paragraphs."}, )


    exist_output = os.listdir(result_path)
    todo_files = [txt_file for txt_file in os.listdir(txt_out_path) if str(Path(txt_file).with_suffix('.json')) not in exist_output]

    inputs = zip([txt_out_path]*len(todo_files),todo_files,[result_path]*len(todo_files),[prompt_conv]*len(todo_files))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = pool.map(process_theory, inputs)
        wait(futures)

def chat_idea_txt(txt_out_path,result_path):

    prompt_conv = [
        {"role": "system",
         "content": ""
         },
    ]
    # authority information
    with open('data/prompt/abstract2idea/system_prompt.txt', 'r') as fi:
        prompt_conv[0]["content"] = fi.read()

    # wrong example and correction
    with open('data/prompt/abstract2idea/208501_example.txt', 'r') as fi:
        prompt_conv.append({'role': 'user', 'content': fi.read()})
    with open('data/prompt/abstract2idea/208501_wrong_result.txt', 'r') as fi:
        prompt_conv.append({'role': 'assistant', 'content': fi.read()})
    with open('data/prompt/abstract2idea/208501_true_result.txt', 'r') as fi:
        prompt_conv.append({'role': 'user', 'content': fi.read()})
    prompt_conv.append({"role": "assistant",
                        "content": "Sorry, I misunderstood your task. I will give you the main idea in the first person as the author(s)."}, )

    exist_output = os.listdir(result_path)
    todo_files = [txt_file for txt_file in os.listdir(txt_out_path) if txt_file not in exist_output]
    
    # process_abstract2idea((txt_out_path,todo_files[0],result_path,prompt_conv))
    
    inputs = zip([txt_out_path]*len(todo_files),todo_files,[result_path]*len(todo_files),[prompt_conv]*len(todo_files))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = pool.map(process_abstract2idea, inputs)
        wait(futures)



def main():
    openai.api_key = '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'
    args = get_args()
    mmd_out_path = Path(args.path)/Path('HT')
    txt_out_path = Path(args.path)/Path('abstract')
    result_path = Path(args.path)/Path('abstract2idea')
    os.makedirs(result_path,exist_ok=True)
    chat_idea_txt(txt_out_path,result_path)

if __name__ == '__main__':
    main()
