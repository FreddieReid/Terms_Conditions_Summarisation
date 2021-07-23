#BERT

#TransformerSum. Use transformers for extractive summarization

# download github

! git clone https://github.com/HHousen/transformersum.git
    
# change cd to trasnformersum

import os

os.chdir("./transformersum")


# run installation
!conda env create --file environment.yml

#activate installation
!conda activate transformersum
! python -m spacy download en_core_web_sm


# download some dependencies
!pip install torch_optimizer

#distilbert-base-uncased-ext-sum Yiu

os.chdir("/kaggle/working/transformersum/src")


from extractive import ExtractiveSummarizer

os.chdir("/kaggle/working/")

model = ExtractiveSummarizer.load_from_checkpoint("../input/distillbertsumext/epoch3.ckpt")


def distillbert_summarizer(x):
    distillbert_summary = ''.join(model.predict(x, num_summary_sentences=1))
    return distillbert_summary


distillbert_ext_summaries_test = test_data["original_text"].map(distillbert_summarizer)
distillbert_ext_summaries_val = validation_data["original_text"].map(distillbert_summarizer)

print("ROUGE 1 SCORE: ",rouge.compute(predictions=distillbert_ext_summaries_test, references=test_data["extractive_summary"], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=distillbert_ext_summaries_test, references=test_data["extractive_summary"], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=distillbert_ext_summaries_test, references=test_data["extractive_summary"], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=distillbert_ext_summaries_test, references=test_data["extractive_summary"])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )


#BERT (Miller)

!pip install bert-extractive-summarizer
!pip install spacy
!pip install transformers # > 2.2.0
!pip install neuralcoref

!python -m spacy download en_core_web_md

from summarizer import Summarizer,TransformerSummarizer

bert_model = Summarizer()

def bert_summarizer(x):
    bert_summary = ''.join(bert_model(x, num_sentences = 1))
    return bert_summary
    
bert_ext_summaries_test = test_data["original_text"].map(bert_summarizer)

#split so only one sentence
df_test = pd.DataFrame(bert_ext_summaries_test).reset_index(drop = True)
bert_ext_summaries_test = df_test['original_text'].str.split(".", expand = True, n=1).reset_index(drop = True)[0]

# get readability scores

!pip install py-readability-metrics
!python -m nltk.downloader punkt
from readability import Readability



from readability import Readability


r = Readability(bert_data_string)
fk = r.flesch_kincaid()
fk.grade_level


cl = r.coleman_liau()
cl.grade_level

dc = r.dale_chall()
dc.score

dc = r.dale_chall()    
dc.grade_levels




