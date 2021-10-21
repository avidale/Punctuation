import transformers

MODEL_SRC = 'cointegrated/rubert-base-lesha17-punctuation'
MODEL_DST = 'model'

model = transformers.AutoModelForTokenClassification.from_pretrained(MODEL_SRC)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_SRC)

model.save_pretrained(MODEL_DST)
tokenizer.save_pretrained(MODEL_DST)

print('The model successfully cached!')
