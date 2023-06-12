from flask_cors import CORS
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse   
from fastapi import FastAPI, File, UploadFile, HTTPException  
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import uvicorn
import os  
import json
import pickle
import pandas as pd
import collections
import time
import numpy as np
import os
import math
import datetime
import tqdm
import transformers
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",  # Update with the appropriate port for your frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



classifier = None
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)


#def newsapi(params): 
def newsapi(stockName):    

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": stockName,#Apple
        "from": "2023-06-07",
        "sortBy": "popularity",
        "apiKey": "e57cdb6ec34943ea8917c0e520fce4cc"
    }

    val = []
    # Send the GET request
    response = requests.get(url, params=params)
    #response = requests.get(url)#, params=params)
    #print(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Process the response data (in JSON format)
        data = response.json()
        
        # Extract and print the articles
        articles = data.get("articles", [])
        for article in articles:
            description = article.get("description", "")
            val.append(description)
            print(val)
    else:
        print(f"Request failed with status code: {response.status_code}")

    return val


weightageList = []

def calculate_average(lst):
    if len(lst) == 0:
        return 0  # Return 0 if the list is empty to avoid division by zero error
    else:
        return sum(lst) / len(lst)

from fastapi import FastAPI, Request
@app.post('/generateSentiment', response_class=ORJSONResponse) 
async def generateSentiment(stockName: str):
    global classifier

    # data = await request.json()
    # stockName = data.get('stockName')
    
    # global classifier
    
    print("Calculatin Sentiment for : ", stockName)
    newsList = newsapi(stockName)
    #newsList = ['Amid a bunch of new products']

    for news in newsList:        
        results = classifier(news)
        # results = classifier("""When it comes to AI-related stocks — and tech stocks in general — Apple is really the beginning and the end of the discussion. After shunning tech stocks for decades, Buffett made a major shift on May 16, 2016, picking up nearly 10 million shares of Apple at a split-adjusted price of about $6.81. Now trading at about $171 per share, Apple has posted huge gains on Buffett’s original purchase. Although Buffett has occasionally trimmed Berkshire’s Apple position, it now sits at a whopping 47% of the company’s entire investment portfolio. Take Our Poll: Who Has Given You the Best Money Advice You Have Ever Received? Regardless of what other AI stocks Berkshire might dabble in, it’s clear that the company’s big bet is on Apple.

        # Buffett has said he views Apple not as a tech company but as a consumer products business. In that field, Buffett sees Apple as the best of the best. In fact, at the Berkshire Hathaway annual shareholders meeting in May 2023, Buffett said Apple “just happens to be a better business than any we own.” Regarding Apple’s iPhone, Buffett said that if people “had to give up a second car or give up their iPhone, they give up their second car.”

        # Apple already has begun integrating AI into its product line. CEO Tim Cook said AI will eventually “affect every product and service we have.” Two of the company’s signature products, the iPhone and the Apple Watch, already use an AI-powered crash detection feature, and the Apple Watch also has an electrocardiogram feature.

        # In its quest to continue its dominance in the consumer products world, Apple seems poised to rapidly integrate AI into its stable”""")
                        
        sentimentVal = round(results[0]['score'] * 10, 2)   
        print(sentimentVal)                                                                                                                                             
        weightageList.append(sentimentVal)

    return calculate_average(weightageList)
    
if __name__ == '__main__':
    
    #global classifier


    uvicorn.run(app ,host="0.0.0.0", port=5000)
