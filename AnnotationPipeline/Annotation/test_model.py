import spacy

nlp = spacy.load("<PATH_TO_TRAINED_MODEL>")

input_string = ""
doc = nlp(input_string)

print(doc.cats)
