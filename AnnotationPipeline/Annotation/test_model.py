import spacy

nlp = spacy.load("./test-model/model-last")

doc = nlp("För mig är svaret idiot enkelt Men om det inte \
 är idiot enkelt så bekräftar det bara hur sorgligt samhället")

print(doc.cats)
