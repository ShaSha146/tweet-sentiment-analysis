# Adapted from code for a Google ColaB Notebook GPU instance

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

########################################################################################################################

df = pd.read_csv("train.csv")

sns.countplot(df.sentiment)
plt.xlabel('sentiments')
plt.show()


def score_conversion(rating):  # Convert all sentiments/ratings to integers to load into model easier
  if rating == 'negative':
    return 0
  elif rating == 'neutral':
    return 1
  else:
    return 2


df['score'] = df.sentiment.apply(score_conversion)  # Append the new data form into the dataset


########################################################################################################################

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'  # Choose the pre-trained load from huggingface libary

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)  # Set the tokenizer to the BERT default tokenizer

token_lens = []

for txt in df.content:  # Get the lengths of each tokenized text input
  tokens = tokenizer.encode(txt, max_length=512, truncation=True)
  token_lens.append(len(tokens))

print(max(token_lens))

MAX_LEN = max(token_lens) + 1  # Get the length of the longest token to use a fixed length of token for BERT

########################################################################################################################


class Data_In(Dataset):  # Encodes data

    def __init__(self, reviews, targets, tokenizer, max_len):  # Just imports the input arguments into the class
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):  # Length of the data
        return len(self.reviews)

    def __getitem__(self, item):  # Gets data encoding from the data
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(review, add_special_tokens=True, max_length=self.max_len,  # Calculates the content encodings
                                              return_token_type_ids=False, pad_to_max_length=True,
                                              return_attention_mask=True, return_tensors='pt',)

        return {'review_text': review, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(), 'targets': torch.tensor(target, dtype=torch.long)}


df_train, df_val = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)  # Split datasets into validation and training
df_test = pd.read_csv("evaluate.csv")  # Load the evaluation dataset

zz = df.score  # Get error using "df.score.to_numpy()" used this placeholder


def load_data(df, tokenizer, max_len, batch_size):  # Function to easily load encoded data

  df_en = Data_In(reviews=df.content.to_numpy(), targets=zz.to_numpy(), tokenizer=tokenizer, max_len=max_len)

  return DataLoader(df_en, batch_size=batch_size, num_workers=4)

########################################################################################################################


BATCH_SIZE = 16  # Increased due to time limitations

train_data_loader = load_data(df_train, tokenizer, MAX_LEN, BATCH_SIZE)  # Actually loads encoded data for each dataset
val_data_loader = load_data(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = load_data(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)  # Loads selected model


class Classifier(nn.Module):  # Runs training on model and uses a dropout layer for regularization and a fully-connected layer for our output.

    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)

        return self.out(output)


classes = ['negative', 'neutral', 'positive']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Moves running of the model to the gpu else cpu

model = Classifier(len(classes))
model = model.to(device)  # Send to assigned device

EPOCHS = 1  # Reduced due to system limitations

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)  # Gets the ADAM optimizer that was recommended for BERT
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  # Use a linear optimizer with no warmup

loss_fn = nn.CrossEntropyLoss().to(device)  # Using CrossEntrophy because this is a classification problem

########################################################################################################################

if __name__ == '__main__':  # Don't remove requirment for other code
    def run_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):  # Function to run model for each epoch

        model = model.train()

        losses = []
        correct_sentiment = 0  # Set intial prediction to zero so the model can only go up with each addition

        for d in data_loader:

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_sentiment += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_sentiment.double() / n_examples, np.mean(losses)


    def run_eval(model, data_loader, loss_fn, device, n_examples): # Function to evaluate the model performance using data loader

      model = model.eval()

      losses = []
      correct_sentiment = 0

      with torch.no_grad():

        for d in data_loader:

          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)

          outputs = model(input_ids=input_ids, attention_mask=attention_mask)
          _, preds = torch.max(outputs, dim=1)

          loss = loss_fn(outputs, targets)

          correct_sentiment += torch.sum(preds == targets)
          losses.append(loss.item())

      return correct_sentiment.double() / n_examples, np.mean(losses)


    history = defaultdict(list)
    best_accuracy = 0  # Set to zero so the accuracy of the model can only go up

    for epoch in range(EPOCHS):  # Training loops

      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)

      train_acc, train_loss = run_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))

      print(f'Train loss {train_loss} accuracy {train_acc}')

      val_acc, val_loss = run_eval(model, val_data_loader, loss_fn, device, len(df_val))

      print(f'Val   loss {val_loss} accuracy {val_acc}')
      print()

      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)

      if val_acc > best_accuracy:

        torch.save(model.state_dict(), 'best_model_state.bin')  # Saves the best model config
        best_accuracy = val_acc

    # Plot the metrics while training (NOT NEEDED FOR ONE EPOCH)
    # plt.plot(history['train_acc'], label='train accuracy')
    # plt.plot(history['val_acc'], label='validation accuracy')
    #
    # plt.title('Training history')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.ylim([0, 1]);

    test_acc, _ = run_eval(model, test_data_loader, loss_fn, device, len(df_test))

    print(test_acc.item())


    def get_predictions(model, data_loader):

        model = model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():

            for d in data_loader:

                texts = d["review_text"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values


    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

    print(classification_report(y_test, y_pred, target_names=classes))


    def show_confusion_matrix(confusion_matrix):   # Plot confusion matrix
      hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
      hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
      hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
      plt.ylabel('True sentiment')
      plt.xlabel('Predicted sentiment');

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    show_confusion_matrix(df_cm)






