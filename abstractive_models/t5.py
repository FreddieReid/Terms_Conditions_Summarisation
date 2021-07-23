# load bleurt
!pip install --upgrade fsspec
!pip install datasets
!pip install rouge_score
from datasets import load_metric 
import rouge_score
rouge = load_metric("rouge")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# install packages for bluert
!pip install --upgrade pip # ensures that pip is current 
!pip install git+https://github.com/google-research/bleurt.git
    
# ensure it is using the cpu to save ram on GPU

bleurt = load_metric("bleurt")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import T5Tokenizer, T5ForConditionalGeneration
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

get_ipython().ast_node_interactivity = 'all'

#get own data

cols1 = ['document_ref', 'original_text', 'abtractive_summary']

own_data = pd.read_csv('../input/summariesf/terms_summarisations.csv', usecols = cols1)

#ManorLi data

cols = ["document_ref", "original_text", "abtractive_summary"]

data = pd.read_csv('../input/summariesf/manor_li_data.csv', usecols = cols)

total_data = pd.concat([data, own_data])

own_data_imp = own_data[own_data["abtractive_summary"] != "not important"]

#split data into training+test

training_data = data
validation_data = own_data_imp[(own_data_imp["document_ref"] == "barc_pc") | (own_data_imp["document_ref"] == "sant_cc_tc")]
test_data = own_data_imp[(own_data_imp["document_ref"] != "barc_pc") & (own_data_imp["document_ref"] != "sant_cc_tc")]

columns = ["original_text", "abtractive_summary"]

training_data = pd.DataFrame(training_data, columns = columns).reset_index(drop=True)
validation_data = pd.DataFrame(validation_data, columns= columns).reset_index(drop=True)
test_data = pd.DataFrame(test_data, columns = columns).reset_index(drop=True)


training_data.columns = ["text", "ctext"]
validation_data.columns = ["text", "ctext"]
test_data.columns = ["text", "ctext"]


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt', truncation = True)
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt', truncation = True)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
      
      
 
def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _%50==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()
        
        
 

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


# WandB – Initialize a new run
wandb.init(project="transformers_tutorials_summarization")

# WandB – Config is a variable that holds and saves hyperparameters and inputs
# Defining some key variables that will be used later on in the training  
config = wandb.config          # Initialize config
config.TRAIN_BATCH_SIZE = 7    # input batch size for training (default: 64)
config.VALID_BATCH_SIZE = 7    # input batch size for testing (default: 1000)
config.TRAIN_EPOCHS = 3        # number of epochs to train (default: 10)
config.VAL_EPOCHS = 1 
config.LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
config.SEED = 42               # random seed (default: 42)
config.MAX_LEN = 512
config.SUMMARY_LEN = 150


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(config.SEED) # pytorch random seed
np.random.seed(config.SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

# tokenzier for encoding the text
tokenizer = T5Tokenizer.from_pretrained("t5-base")


# Creating the Training and Validation dataset for further creation of Dataloader
training_set = CustomDataset(training_data, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
val_set = CustomDataset(validation_data, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': config.TRAIN_BATCH_SIZE,
    }

val_params = {
    'batch_size': config.VALID_BATCH_SIZE,
    }


# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)



# Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
# Further this model is sent to device (GPU/TPU) for using the hardware.
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model = model.to(device)

# Defining the optimizer that will be used to tune the weights of the network in the training session. 
optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

# Log metrics with wandb
wandb.watch(model, log="all")
# Training loop
print('Initiating Fine-Tuning for the model on our dataset')

for epoch in range(config.TRAIN_EPOCHS):
    train(epoch, tokenizer, model, device, training_loader, optimizer)


# Validation loop and saving the resulting file with predictions and acutals in a dataframe.
# Saving the dataframe as predictions.csv
print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
for epoch in range(config.VAL_EPOCHS):
    predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
    final_df = pd.DataFrame(predictions, actuals).reset_index()
    #final_df.to_csv('./models/predictions.csv')
    print('Output Files generated for review')


#55351584650c1776ec7b4204c92ad0049d629818

#t5 base with batch size 7, 3 epcochs use 14.1


print("ROUGE 1 SCORE: ",rouge.compute(predictions=final_df["index"], references=final_df[0], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=final_df["index"], references=final_df[0], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=final_df["index"], references=final_df[0], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=final_df["index"], references=final_df[0])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )



test_params = {
    'batch_size': config.VALID_BATCH_SIZE,
    }


test_set = CustomDataset(test_data, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
test_loader = DataLoader(test_set, **test_params)


for epoch in range(config.VAL_EPOCHS):
    predictions, actuals = validate(epoch, tokenizer, model, device, test_loader)
    final_df_test = pd.DataFrame(predictions, actuals).reset_index()
    #final_df.to_csv('./models/predictions.csv')
    print('Output Files generated for review')
    
    
print("ROUGE 1 SCORE: ",rouge.compute(predictions=final_df_test["index"], references=final_df_test[0], rouge_types=["rouge1"])["rouge1"].mid)
print("ROUGE 2 SCORE: ",rouge.compute(predictions=final_df_test["index"], references=final_df_test[0], rouge_types=["rouge2"])["rouge2"].mid)
print("ROUGE L SCORE: ",rouge.compute(predictions=final_df_test["index"], references=final_df_test[0], rouge_types=["rougeL"])["rougeL"].mid)
bleurt_scores = bleurt.compute(predictions=final_df_test["index"], references=final_df_test[0])
print("BLEURT SCORE: ", sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"]) )
