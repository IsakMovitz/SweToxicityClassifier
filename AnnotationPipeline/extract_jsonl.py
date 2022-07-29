import re
import random
import json

span_length = 20

def extract_keyword_json(input_file,output_file,thread):
    #Keywords
    keywords_list = []
    with open('keywords.txt',encoding="utf-8") as keyword_file:
        for line in keyword_file:
            word = line.strip('\n')
            keywords_list.append(word)

    id_nr = 0

    with open(input_file) as current_file:

        with open(output_file, 'w', encoding='utf-8') as f:
            for line in current_file:       # Right now only taking one sentence from each thread.

                random.shuffle(keywords_list)
        
                # The first keyword encountered
                for word in keywords_list:
                    
                    separate_word = " " + word + " "

                    if separate_word in line:
                        line_list = re.split(r'\\n|\n| ' , line)
                        line_list = [x for x in line_list if not x.endswith(":")]
    
                        index = line_list.index(word)

                        # Randomized span where the word is included
                        start_nr = random.randint(1,7)
                        end_nr = span_length - start_nr
                        start_index = index - start_nr
                        extracted_sentence = line_list[start_index:index + end_nr]

                        str1 = " " 
                        text_sample = str1.join(extracted_sentence)

                        jsonl_line = {"thread":thread ,"id":id_nr,"text":text_sample,"keyword": separate_word ,"starting_index":start_index,"span_length":span_length}

                        f.write(json.dumps(jsonl_line,ensure_ascii=False) + "\n")

                        break

        
                id_nr += 1

antisemitism_sionism_och_judiska_maktforhallanden_thread = "rest-arkiverade_forum-antisemitism_sionism_och_judiska_maktforhallanden"
extract_keyword_json("./antisemitism_sionism_och_judiska_maktforhallanden.txt","./keyword_antisemitism_sionism_och_judiska_maktforhallanden.jsonl",antisemitism_sionism_och_judiska_maktforhallanden_thread)

def clean_data(input_file, output_file):

    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
        with open(output_file, 'w', encoding='utf-8') as f:

     
            for json_str in json_list:
                result = json.loads(json_str)
      
                text = result['text'].split(" ")

                # Removing weirdness with spaces being in lists, now all spans will be 15
                while '' in text:
                    text.remove('')

    
                if len(text) >= span_length:
                    
                    f.write(json.dumps(result,ensure_ascii=False) + "\n")


#clean_data("./sampled_20_data/random_20span_500each.jsonl","./sampled_20_data/clean_random_20span_500each.jsonl")

def createDataset(input_txt,keyword_name,keyword_final_name,random_name,random_final_name, thread):


    #extract_keyword_json(input_txt,keyword_name, thread)           
    extract_random_json(input_txt,random_name, thread)
    #clean_data(keyword_name,keyword_final_name)
    clean_data(random_name,random_final_name)


def reformat_final_data(input_file, output_file):
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            
            it = 4000
            for json_str in json_list:
                result = json.loads(json_str)

                thread = result["thread"]
                id_nr = result["id"]
                text_sample = result["text"]
                #keyword = result['keyword']
                starting_index = result["starting_index"]
                span_length = result["span_length"]


                jsonl_line = {"id": it,"thread":thread ,"thread_id":id_nr,"text":text_sample,"starting_index":starting_index,"span_length":span_length}

                f.write(json.dumps(jsonl_line,ensure_ascii=False) + "\n")

                it += 1


#reformat_final_data("./sampled_20_data/random_20span_500each.jsonl","./sampled_20_data/reformat_random_20span_500each.jsonl")

#clean_data("./keyword_data/flashback_keyword_data.jsonl", "./keyword_data/flashback_keyword_data.jsonl")

#reformat_final_data("./keyword_data/flashback_keyword_data.jsonl","./keyword_data/reformatted_clean_flashback_keyword_data.jsonl")

import random

def shuffle_jsonl(input_file,output_file):
  
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
        random.shuffle(json_list)
        with open(output_file, 'w', encoding='utf-8') as f:

            for json_str in json_list:
                result = json.loads(json_str)
    
                f.write(json.dumps(result,ensure_ascii=False) + "\n")


#shuffle_jsonl("./sampled_20_data/reformat_random_20span_500each.jsonl","./sampled_20_data/shuffled_random_20span_dataset.jsonl")