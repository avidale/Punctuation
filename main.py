import os
from typing import Optional

import transformers
import torch
import re

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

LABELS = ' !,-.:;?'

model = transformers.AutoModelForTokenClassification.from_pretrained('model')
tokenizer = transformers.AutoTokenizer.from_pretrained('model')
if torch.cuda.is_available():
    model.cuda()


def restore_punct_on_chunk(text):
    encoded = tokenizer(text, max_length=512, truncation=True, return_tensors='pt', return_offsets_mapping=True)
    offsets = encoded['offset_mapping'][0].tolist()
    encoded = {k: v.to(model.device) for k, v in encoded.items() if k != 'offset_mapping'}
    with torch.inference_mode():
        out = model(**encoded).logits[:, :, :len(LABELS)].argmax(-1).cpu().numpy()[0]

    # restore the text from the original spans and the predicted punctuation
    prev_b = 0
    prev_e = 0
    spans = []
    for i, pun in enumerate(out):
        b, e = offsets[i]
        if b < prev_e:
            continue
        # restore the spaces
        if b > prev_e:
            spans.append(text[prev_e: b])
        # restore the original words
        spans.append(text[b:e])
        # add the punctuation
        if pun != 0:
            if LABELS[pun] == '-':
                spans.append(' ')  # because there will be another space after
            spans.append(LABELS[pun])
        prev_b, prev_e = b, e
    spans.append(text[prev_e:])
    return ''.join(spans)


def split_text(text, max_chunk_size = 3000, min_chunk_size = 1000):
    big_chunks = [text[s.start():s.end()] for s in re.finditer('[^\n]*?(\n+|$)', text)]
    chunks = []
    for big_chunk in big_chunks:
        if len(big_chunk) <= max_chunk_size:
            chunks.append(big_chunk)
        else:
            while len(big_chunk) >= max_chunk_size:
                position = big_chunk.find(' ', min_chunk_size)
                chunks.append(big_chunk[:position])
                big_chunk = big_chunk[position:]
            chunks.append(big_chunk)
    return chunks


def restore_punct(text):
    new_chunks = []
    for chunk in split_text(text):
        if chunk.strip():
            new_chunks.append(restore_punct_on_chunk(chunk))
        else:
            new_chunks.append(chunk)
    return ''.join(new_chunks)


app = FastAPI()


class Text(BaseModel):
    """
    `text` is the required field.
    `restored` is the output field that contains the restored text.
    """
    text: str
    restored: Optional[str] = None


@app.post("/restore-punct")
async def process_text(item: Text):
    item.restored = restore_punct(item.text)
    return item


if __name__ == '__main__':
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host=host, port=port)
