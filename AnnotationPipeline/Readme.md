* Annotation pipeline

** Pre-processing

Pre-processing was done by using jq to extract segments from a larger web-scraped json-file from the Swedish commenting forum Flashback. Here the file 
"antisemitism_sionism_och_judiska_maktforhallanden.json" is used as an example to showcase the process.

- jq '.text' ./subtopics_files/antisemitism_sionism_och_judiska_maktforhallanden.json > ./sampled_data/antisemitism_sionism_och_judiska_maktforhallanden.txt

Text spans were then extracted by using keywords into a jsonl format with the script "extract_jsonl.py" in order to create a file of unlabelled data.

** Annotation 

Annotation was done using the annotation tool Prodigy and incorporating Active learning. Examples can be seen in the Data-folder. 