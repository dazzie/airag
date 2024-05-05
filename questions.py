# questions.py
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List
from scipy import spatial
from dotenv import load_dotenv

import os

load_dotenv()


def distances_from_embeddings(
  query_embedding: List[float],
  embeddings: List[List[float]],
  distance_metric="cosine",
) -> List[List]:
  """Return the distances between a query embedding and a list of embeddings."""
  distance_metrics = {
      "cosine": spatial.distance.cosine,
      "L1": spatial.distance.cityblock,
      "L2": spatial.distance.euclidean,
      "Linf": spatial.distance.chebyshev,
  }
  distances = [
      distance_metrics[distance_metric](query_embedding, embedding)
      for embedding in embeddings
  ]
  return distances


openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def answer_question(df,
                    model="gpt-3.5-turbo-1106",
                    question="What is the meaning of life?",
                    max_len=1800,
                    debug=False,
                    max_tokens=150,
                    stop_sequence=None):
  """
    Answer a question based on the most similar context from the dataframe texts
  """
  context = create_context(
      question,
      df,
      max_len=max_len,
  )
  # If debug, print the raw model response
  if debug:
    print("Context:\n" + context)
    print("\n\n")

  try:
    # Create a completions using the question and context
    response = openai.chat.completions.create(
        model=model,
        messages=[{
            "role":
            "user",
            "content":
            f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know.\" Try to site sources to the links in the context when possible.\n\nContext: {context}\n\n---\n\nQuestion: {question}\nSource:\nAnswer:",
        }],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop_sequence,
    )
    return response.choices[0].message.content
  except Exception as e:
    print(e)
    return ""
