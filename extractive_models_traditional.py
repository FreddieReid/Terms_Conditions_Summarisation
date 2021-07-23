!pip install sumy
import sumy
from sumy import summarizers

#TextRank

# Importing the parser and tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

#tokenize

def parser(x):
    parser = PlaintextParser(x,Tokenizer('english'))
    return parser


textrank_parsed_val = validation_data["original_text"].map(parser)

#summarizer

summarizer = TextRankSummarizer()

def textrank_summarizer():
    summaries = []
    for clause in textrank_parsed_val:
        summs = summarizer(clause.document,1)
        summaries.append(summs)
    return pd.DataFrame(summaries)
    
textrank_summaries_val = textrank_summarizer()

#turn sentence objects into strings

textrank_summaries_val["textrank_summary"] = textrank_summaries_val[0].apply(lambda x: str(x))

#delete unnneeded column

textrank_summaries_val.drop(0, axis = 1, inplace = True)


print("ROUGE 1 SCORE: ",rouge.compute(predictions=textrank_summaries_val["textrank_summary"]  , references=validation_data["extractive_summary"], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=textrank_summaries_val["textrank_summary"]  , references=validation_data["extractive_summary"], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=textrank_summaries_val["textrank_summary"]  , references=validation_data["extractive_summary"], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=textrank_summaries_val["textrank_summary"]  , references=validation_data["extractive_summary"])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )


#Lex Rank
# Importing the parser and tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#tokenize

def parser(x):
    parser = PlaintextParser(x,Tokenizer('english'))
    return parser


lexrank_parsed_val = validation_data["original_text"].map(parser)

#summarizer

summarizer = LexRankSummarizer()

def lexrank_summarizer():
    summaries = []
    for clause in lexrank_parsed_val:
        summs = summarizer(clause.document,1)
        summaries.append(summs)
    return pd.DataFrame(summaries)
    
lexrank_summaries_val = lexrank_summarizer()

#turn sentence objects into strings

lexrank_summaries_val["lexrank_summary"] = lexrank_summaries_val[0].apply(lambda x: str(x))

#delete unnneeded column

lexrank_summaries_val.drop(0, axis = 1, inplace = True)

#LSA

from sumy.summarizers.lsa import LsaSummarizer


lsa_parsed = validation_data["original_text"].map(parser)

summarizer=LsaSummarizer()

def lsa_summarizer():
    summaries = []
    for clause in lsa_parsed:
        summs = summarizer(clause.document,1)
        summaries.append(summs)
    return pd.DataFrame(summaries)

lsa_summaries_val = lsa_summarizer()

#turn sentence objects into strings

lsa_summaries_val["lsa_summary"] = lsa_summaries_val[0].apply(lambda x: str(x))

#delete unnneeded column

lsa_summaries_val.drop(0, axis = 1, inplace = True)


#Luhn

from sumy.summarizers.luhn import LuhnSummarizer


luhn_parsed_val = validation_data["original_text"].map(parser)

summarizer=LuhnSummarizer()

def luhn_summarizer():
    summaries = []
    for clause in luhn_parsed_val:
        summs = summarizer(clause.document,1)
        summaries.append(summs)
    return pd.DataFrame(summaries)

luhn_summaries_val = luhn_summarizer()

#turn sentence objects into strings

luhn_summaries_val["luhn_summary"] = luhn_summaries_val[0].apply(lambda x: str(x))

#delete unnneeded column

luhn_summaries_val.drop(0, axis = 1, inplace = True)

#KLSum

from sumy.summarizers.kl import KLSummarizer


kl_parsed = test_data["original_text"].map(parser)

summarizer=KLSummarizer()

def kl_summarizer():
    summaries = []
    for clause in kl_parsed:
        summs = summarizer(clause.document,1)
        summaries.append(summs)
    return pd.DataFrame(summaries)

kl_summaries_test = kl_summarizer()

#turn sentence objects into strings

kl_summaries_test["kl_summary"] = kl_summaries_test[0].apply(lambda x: str(x))

#delete unnneeded column

kl_summaries_test.drop(0, axis = 1, inplace = True)


#Baseline


test_data["baseline_extractive"] = test_data["original_text"].str.split(".", expand = True)[0]
validation_data["baseline_extractive"] = validation_data["original_text"].str.split(".", expand = True)[0]

print("ROUGE 1 SCORE: ",rouge.compute(predictions=validation_data["baseline_extractive"] , references=validation_data["extractive_summary"], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=validation_data["baseline_extractive"]  , references=validation_data["extractive_summary"], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=validation_data["baseline_extractive"]  , references=validation_data["extractive_summary"], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=validation_data["baseline_extractive"] , references=validation_data["extractive_summary"])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )
