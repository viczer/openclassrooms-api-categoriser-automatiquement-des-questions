from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from .utils import clean_html, text_cleaning, tokenize, filtering_nouns, lemmatize, LdaModel, SupervisedModel

app = FastAPI(title='API for tags prediction on Stack Overflow posts',
              description='Return tags related to a poste',
              version='0.0.1')

@app.get("/")
def root():
    return {"Welcome to the API. Check /docs for usage"}

class Input(BaseModel):
    text : str

@app.post("/predict")
async def get_prediction(data: Input):

    text_wo_html = clean_html(data.text)
    cleaned_text = text_cleaning(text_wo_html)
    tokenized_text = tokenize(cleaned_text)
    filtered_noun_text = filtering_nouns(tokenized_text)
    lemmatized_text = lemmatize(filtered_noun_text)
    lda_model = LdaModel()
    unsupervised_pred = lda_model.predict_tags(lemmatized_text)
    supervised_model = SupervisedModel()
    supervised_pred = supervised_model.predict_tags(lemmatized_text)
    text = jsonable_encoder(data.text)

    return JSONResponse(status_code=200, content={"text": text, 
                                                  "unsupervised_tags": unsupervised_pred,  
                                                  "supervised_tags": supervised_pred})

