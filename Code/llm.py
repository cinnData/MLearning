## Hugging face models ##

# Resources #
import pandas as pd
from transformers import pipeline

# Sentiment analysis #
task = 'text-classification'
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
classify = pipeline(task, model=model_name)
prompt = 'The food was good, but service at the restaurant was a bit slow'
output = classify(prompt)
pd.DataFrame(output)

# Zero-shot classification #
task = 'zero-shot-classification'
model_name = 'facebook/bart-large-mnli'
classify = pipeline(task, model=model_name)
prompt = 'I have a problem with my iphone that needs to be resolved asap!'
labels = ['urgent', 'not urgent']
output = classify(prompt, candidate_labels=labels)
pd.DataFrame(output)

# Text generation #
task = 'text-generation'
model_name = 'gpt2'
generate = pipeline(task, model=model_name)
prompt = 'The Gion neighborhood in Kyoto is famous for'
output = generate(prompt, max_length=100)
pd.DataFrame(output)

# Summarization #
task = 'summarization'
model_name = 'facebook/bart-large-cnn'
summarize = pipeline(task, model=model_name)
prompt =  """Walking amid Gion's Machiya wooden houses is a mesmerized experience.
	The beautifully preserved structures exuded an old world charm that transports
	visitors back in time, making them feel like they have stepped into a living
	museum. The glow of lanterns lining the narrow streets add to the enchanting
	ambiance, making each stroll a memorable journey through Japan's rich cultural
	history."""
output = summarize(prompt, max_length=60, clean_up_tokenization_spaces=True)
print(output[0]['summary_text'])

# Question answering #
task = 'question-answering'
model_name = 't5-small'
answer = pipeline(task, model=model_name)
context =  "Walking amid Gion's Machiya wooden houses was a mesmerized experience."
question = 'What are Machiya houses made of?'
output = answer(question, context)
print(output['answer'])

# Translation #
task = 'translation_en_to_es'
model_name = 'Helsinki-NLP/opus-mt-en-es'
translate  = pipeline(task, model=model_name)
text =  "Walking amid Gion's Machiya wooden houses was a mesmerized experience."
output = translate(text, clean_up_tokenization_spaces=True)
print(output[0]['translation_text'])

