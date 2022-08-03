from functions import *
import json

def find_wrongly_classified(input_filename, output_filename):
    cleaned_data = []
    finetuned_model = AutoModelForSequenceClassification.from_pretrained("<PATH_TO_TRAINED_MODEL>")
    tokenizer = AutoTokenizer.from_pretrained("<PATH_TO_TRAINED_MODEL>")

    # Open annotated file:
    with open(input_filename, 'r',encoding='utf-8') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        line_dict = json.loads(json_str)
        text_sample = line_dict['text']
        toxic = line_dict['TOXIC']

        result = evaluate_string(text_sample,finetuned_model, tokenizer)
        if toxic != result:
            cleaned_data.append(line_dict)

    # Save cleaned data back to jsonl format:
    with open(output_filename, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                if item != cleaned_data[-1]:
                    f.write(json.dumps(item,ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(item,ensure_ascii=False))

def classify_string(input_string):

    finetuned_model = AutoModelForSequenceClassification.from_pretrained("<PATH_TO_TRAINED_MODEL>")
    tokenizer = AutoTokenizer.from_pretrained("<PATH_TO_TRAINED_MODEL>")
    result = evaluate_string(input_string, finetuned_model, tokenizer)
    print(result)



if __name__ == '__main__':
    
    input_string = ""

    classify_string(input_string)
    find_wrongly_classified("<INPUT_FILE>","<OUTPUT_FILE>")
    




