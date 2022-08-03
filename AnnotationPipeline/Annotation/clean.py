import json

def cleaner(input_filename, output_filename):
    cleaned_data = []
    id_list = []

    with open(input_filename, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        line_dict = json.loads(json_str)
        id_nr = line_dict['id']
        if id_nr not in id_list:
            id_list.append(id_nr)
            
            text_sample = line_dict['text']
            thread = line_dict['thread']
            thread_id = line_dict['thread_id']
            keyword = line_dict['keyword'].strip(" ")
            starting_index = line_dict['starting_index']
            span_length = line_dict['span_length']

        if line_dict['answer'] == 'accept': # 1
            cleaned_data.append({"id":id_nr,"thread":thread,"thread_id":thread_id, "text":text_sample, "keyword": keyword ,"starting_index":starting_index,
            "span_length":span_length,"TOXIC":1})
        elif line_dict['answer'] == 'reject':
            cleaned_data.append({"id":id_nr,"thread":thread,"thread_id":thread_id,"text":text_sample,"keyword": keyword,"starting_index":starting_index,
            "span_length":span_length,"TOXIC":0})
        else:
            pass

    # Save cleaned back to jsonl format:
    with open(output_filename, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                if item != cleaned_data[-1]:
                    f.write(json.dumps(item,ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(item,ensure_ascii=False))

cleaner("./Data/annotated_data.jsonl","./Data/clean_annotated_data.jsonl")
