
# calculate rouge scores
!pip install rouge_score
!pip install datasets
#!pip install --upgrade fsspec
from datasets import load_metric

rouge = load_metric("rouge")

# install packages for bluert
!pip install --upgrade pip # ensures that pip is current 
!pip install git+https://github.com/google-research/bleurt.git
    
# ensure it is using the cpu to save ram on GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# load bleurt

bleurt = load_metric("bleurt")

# get professionally annotated data


cols = ["doc","original_text", "reference_summary"]

data = pd.read_json('https://raw.githubusercontent.com/lauramanor/legal_summarization/master/all_v1.json').T
data = pd.DataFrame(data, columns = cols).reset_index(drop=True)
data.columns = ["document_ref", "original_text", "abtractive_summary"]

get_ipython().ast_node_interactivity = 'all'

#get own data

cols1 = ['document_ref', 'original_text', 'extractive_summary', 'abtractive_summary']

own_data = pd.read_csv('../input/summary/terms_summarisations.csv', usecols = cols1)

own_data.isnull().sum()

len(own_data)

#select extractive summaries for score testing

own_data_imp = own_data[own_data["extractive_summary"] != "not important"]

own_data_imp.shape
# split 3 (no not important)

training_data = data
validation_data = own_data_imp[(own_data_imp["document_ref"] == "barc_pc") | (own_data_imp["document_ref"] == "sant_cc_tc")]
test_data = own_data_imp[(own_data_imp["document_ref"] != "barc_pc") & (own_data_imp["document_ref"] != "sant_cc_tc")]
training_data.shape
validation_data.shape
test_data.shape

from transformers import pipeline

# Using BART
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization", model = 'sshleifer/distilbart-cnn-12-6')

bart_summaries_test = test_data.original_text.map(summarizer)
bart_summaries_val = validation_data.original_text.map(summarizer)


from transformers import pipeline

# Using BART
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization", model = 'sshleifer/distilbart-cnn-12-6')

bart_summaries_test = test_data.original_text.map(summarizer)
bart_summaries_val = validation_data.original_text.map(summarizer)

# turn dictionary into series
#first exrtact values from dictionary

lis = []
for line in bart_summaries_test:
    lis.append(*line[0].values())
    
#turn list into series

bart_summaries_test = pd.Series(lis)

#get test results

print("ROUGE 1 SCORE: ",rouge.compute(predictions=bart_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=bart_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=bart_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=bart_summaries_test, references=test_data["abtractive_summary"])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )


# Using T5
from transformers import pipeline

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

t5_summaries_test = test_data.original_text.map(summarizer)
t5_summaries_val = validation_data.original_text.map(summarizer)

# turn dictionary into series
#first exrtact values from dictionary

lis = []
for line in t5_summaries_test:
    lis.append(*line[0].values())
    
#turn list into series

t5_summaries_test = pd.Series(lis)

t5_summaries_test

print("ROUGE 1 SCORE: ",rouge.compute(predictions=t5_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=t5_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=t5_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=t5_summaries_test, references=test_data["abtractive_summary"])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )




from transformers import pipeline


summarizer = pipeline("summarization", model= "sshleifer/distill-pegasus-cnn-16-4", tokenizer ="sshleifer/distill-pegasus-cnn-16-4")

#pegasus_summaries_test = test_data.original_text.map(summarizer)
pegasus_summaries_val = validation_data.original_text.map(summarizer)

lis = []
for line in pegasus_summaries_test:
    lis.append(*line[0].values())
    
#turn list into series

pegasus_summaries_test = pd.Series(lis)

pegasus_summaries_test

print("ROUGE 1 SCORE: ",rouge.compute(predictions=pegasus_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=pegasus_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=pegasus_summaries_test, references=test_data["abtractive_summary"], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=pegasus_summaries_test, references=test_data["abtractive_summary"])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )





