!pip install datasets
!pip install --upgrade fsspec
import datasets
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch.nn as nn
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

get_ipython().ast_node_interactivity = 'all'

#get own data

cols1 = ['document_ref', 'original_text', 'extractive_summary', 'abtractive_summary']

own_data = pd.read_csv('../input/summariesd/terms_summarisations.csv', usecols = cols1)

own_data.isnull().sum()

# create binary importance classifier


own_data["labels"] = np.where(own_data['abtractive_summary']=='not important', 0, 1)

round(own_data["labels"].value_counts(normalize = True),3)*100

cols = ["document_ref", "original_text", "labels"]

data = pd.read_csv('../input/manorandlis/manor_li_data.csv', usecols = cols)

total_data = pd.concat([data, own_data])

round(total_data["labels"].value_counts(normalize = True),3)*100

# the split between important and non-important data is 58.6/41.4

# create a baseline value
# P(class is 0) * P(you guess 0) + P(class is 1) * P(you guess 1)

baseline = (0.586 * 0.586) + (0.414 * 0.414)

print("The baseline performance is {:.2f}".format(baseline))


#data split
from sklearn.model_selection import train_test_split
 
X = total_data["original_text"]
y= total_data["labels"]   
    
#split into train and test set with stratification on y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


#split test set into validation and test set

X_val = X_test[:125]
X_test = X_test[125:]
y_val = y_test[:125]
y_test = y_test[125:]

#concatenate text with the labels

training_data = pd.concat([X_train, y_train], axis = 1)
validation_data = pd.concat([X_val, y_val], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)

# create Datsets

training_dataset = Dataset.from_pandas(training_data)
validation_dataset = Dataset.from_pandas(validation_data)
test_dataset = Dataset.from_pandas(test_data)

from transformers import AutoConfig

config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path = 'xlnet-base-cased',
        num_labels=2,
        finetuning_task="text-classification")

from transformers import XLNetTokenizerFast, XLNetForSequenceClassification

tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', use_fast = True)
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', config = config).to("cuda")


from transformers import AutoConfig

config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path = 'bert-base-uncased',
        num_labels=2,
        finetuning_task="text-classification")

from transformers import BertTokenizerFast, BertForSequenceClassification

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = config).to("cuda")
#id2label={id: label for label, id in label2id.items()}

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from transformers import AutoConfig

config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path = 'roberta-base',
        num_labels=2,
        finetuning_task="text-classification")

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', config = config).to("cuda")


def preprocess_function(examples):
    inputs = examples["original_text"]
    model_inputs = tokenizer(inputs, padding= True, truncation=True)
    return model_inputs
  
  #only keep wanted columns

column_names= ["__index_level_0__"]

training_dataset = training_dataset.map(preprocess_function,batched=True,remove_columns = column_names)

eval_dataset = validation_dataset.map(preprocess_function,batched=True,remove_columns = column_names)

test_dataset = test_dataset.map(preprocess_function,batched=True,remove_columns = column_names)

cols = ["original_text"]

training_dataset = training_dataset.remove_columns(cols)
eval_dataset = eval_dataset.remove_columns(cols)
test_dataset = test_dataset.remove_columns(cols)


from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels,preds)
    f1 = f1_score(labels,preds)
    return {
          'accuracy': acc,
          'precision': prec,
          'recall': rec,
        'f1': f1
  }
  
  
  
# define the training arguments

output_dir = 'saved_models'
!mkdir "saved_models/" 

training_args = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs=2,
    per_device_train_batch_size = 15,
    gradient_accumulation_steps = 12,    
    per_device_eval_batch_size= 15,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    weight_decay=0.01,
    logging_steps = 2,
    fp16 = False,
    run_name = 'roberta-classification',
    #push_to_hub = True,
    #push_to_hub_model_id = "roberta_classification",
    #push_to_hub_token = "DozGZHSMVgBAARyDRWLvsgucXGkekbKfWSYAMWLqNLAHznhDMtBHWPMyEdSzCRKHrZrWkdaPOrMuVtXNVdKJylrXOulFQBeACopDwkFRmvQTaukBdRrJlhlKSQkWqpNW"
)


# for roberta uses 15.7 gb ram
# bert uses 15.2


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=training_dataset,
    eval_dataset=eval_dataset
)

%%time

train_result = trainer.train()

trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics

#55351584650c1776ec7b4204c92ad0049d629818

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


predictions = trainer.predict(test_dataset=eval_dataset).predictions
predictions = np.argmax(predictions, axis=1)

model_val_acc = 100 * accuracy_score(test_data.labels.values, predictions)
print("model's test accuracy score is {:.2f}%".format(model_val_acc))

model_val_acc = 100 * precision_score(test_data.labels.values, predictions)
print("model's test precision score is {:.2f}%".format(model_val_acc))

model_val_acc = 100 * recall_score(test_data.labels.values, predictions)
print("model's test recall score is {:.2f}%".format(model_val_acc))

model_val_acc = 100 * f1_score(test_data.labels.values, predictions)
print("model's test f1 score is {:.2f}%".format(model_val_acc))
