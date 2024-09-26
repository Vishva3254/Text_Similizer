# Text similarity
from sentence_transformers import SentenceTransformer, util

def similarity(text1, text2):
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding1, embedding2)

    return similarity.item()


text1 = "The weather is sunny today."
text2 = "It's a bright and sunny day."
print(similarity(text1, text2))