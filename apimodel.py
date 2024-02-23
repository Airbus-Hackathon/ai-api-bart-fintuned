from pydantic import BaseModel
from BART_utilities import *
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
import uvicorn
from fastapi import FastAPI, HTTPException
import pytorch_lightning as pl
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', add_prefix_space=True)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

tldr = pd.read_json('./dataset/airbus_helicopters_train_set.json')
tldr = tldr.transpose().reset_index()
tldr.rename(columns = {'original_text':'source', 'reference_summary':'target'}, inplace = True)
tldr_select = tldr[['source', 'target']]
new_tokens = ['<F>', '<RLC>', '<A>', '<S>', '<P>', '<R>', '<RPC>']

special_tokens_dict = {'additional_special_tokens': new_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
bart_model.resize_token_embeddings(len(tokenizer))
summary_data = SummaryDataModule(tokenizer, tldr_select, batch_size = 1)
model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model)

loaded_model = LitModel.load_from_checkpoint("./models/output.ckpt",  map_location=torch.device("cpu"), learning_rate=2e-5, tokenizer=tokenizer, model=bart_model)


class Text(BaseModel):
    text: str
    
@app.post("/summarize")
async def summarize(request: Text):
    
    try:
        
        inputs = tokenizer(request.text, return_tensors="pt", max_length=512, truncation=True)
        generated_text = loaded_model.generate_text(inputs, eval_beams=5) 
        
        return {"summary": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)