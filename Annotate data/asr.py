import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import librosa
import tqdm
import string
from transformers import Trainer
import pyarabic.araby as araby
import pyarrow
from datasets import load_dataset, load_metric, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, EarlyStoppingCallback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
import optuna
import random
from tabulate import tabulate
import wave
import contextlib

#--------------------------------------------------------------------------------------------------------------------------


df = pd.read_csv("dvoice/darija/texts/train.csv", delimiter="\t")
df_val = pd.read_csv("dvoice/darija/texts/dev.csv", delimiter="\t")
df_test = pd.read_csv("dvoice/darija/texts/test.csv", delimiter="\t")

#--------------------------------------------------------------------------------------------------------------------------

def fullPathData(x):
    return "data/"+x

def fullPathDvoice(x):
    return "dvoice/darija/wavs/"+x

#--------------------------------------------------------------------------------------------------------------------------
df_dirlkit = pd.read_csv("df.csv")
df_dirlkit["wav"]  = df_dirlkit["file"]
df_dirlkit["words"]  = df_dirlkit["text"]

df_dirlkit = df_dirlkit.drop(columns = ["file","text"], axis = 0)

df_dirlkit["wav"] = df_dirlkit["wav"].apply(fullPathData)

#--------------------------------------------------------------------------------------------------------------------------

df["wav"] = df["wav"].apply(fullPathDvoice)
df_val["wav"] = df_val["wav"].apply(fullPathDvoice)
df_test["wav"] = df_test["wav"].apply(fullPathDvoice)
df = df.append(df_dirlkit)
#--------------------------------------------------------------------------------------------------------------------------

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("othrif/wav2vec2-large-xlsr-moroccan", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("othrif/wav2vec2-large-xlsr-moroccan")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#--------------------------------------------------------------------------------------------------------------------------

class AsrData(Dataset):
  def __init__(self, dataset, processor) -> None:
      super().__init__()
      self.audios = []
      letters = list(string.ascii_lowercase)
      numbers = [f"{i}" for i in range (10)]
      duration_total = 0
      chars = ["'", ",", ".", "?", "'", "\"", ";", "|", "-", "_", ":", "!", "@", "#", "$" , "%", "^" , "&" , "*", "(", ")", "+", "=", "\t", "٠","،","\'َ'", "؟", "/"]

      for i in tqdm.tqdm(range(dataset.shape[0])):
        path = df.iloc[i]["wav"]
        try:
          transcript = df.iloc[i]["words"]
          sr = 16000
          y, sr = librosa.load(path, sr = sr)
          
          y = processor(y, sampling_rate=sr).input_values[0]
          
          transcript  = araby.strip_diacritics(transcript)

          for l in letters:
            if l in transcript:
              transcript = transcript.replace(l, "")
          for n in numbers:
            if n in transcript:
              transcript = transcript.replace(n, "")

          for c in chars:
            if c in transcript:
              transcript = transcript.replace(c, "")
              
          with processor.as_target_processor():
              transcript = processor(transcript).input_ids

          if transcript != []:

            self.audios.append({
                    "input_values": y,
                    "labels": transcript,
                })
            
            duration_total += len(y) / sr
        except:
          pass


      print(f"{duration_total/3600} h : {(duration_total % 3600)/60} min")

  
  def __getitem__(self, index):
      return self.audios[index]

  def __len__(self):
        return (len(self.audios))
  
# Define data

data = AsrData(df, processor)
val_data = AsrData(df_val, processor)
test_data = AsrData(df_test, processor)

#--------------------------------------------------------------------------------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
# Define data collator

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

#--------------------------------------------------------------------------------------------------------------------------

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

#--------------------------------------------------------------------------------------------------------------------------

# Define metric calculation

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

#--------------------------------------------------------------------------------------------------------------------------

model = Wav2Vec2ForCTC.from_pretrained(
    "othrif/wav2vec2-large-xlsr-moroccan", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True
)

#--------------------------------------------------------------------------------------------------------------------------

model.freeze_feature_extractor()

#--------------------------------------------------------------------------------------------------------------------------

training_args = TrainingArguments(
  output_dir="Medyassine/xlsr_darija_od",
  group_by_length=True,
  per_device_train_batch_size=10,
  evaluation_strategy="steps",
  num_train_epochs=30,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=1600,
  eval_steps=400,
  logging_steps=400,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
  push_to_hub=True,
  report_to="wandb",
  load_best_model_at_end = True,
)

#--------------------------------------------------------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data,
    eval_dataset=val_data,
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

#--------------------------------------------------------------------------------------------------------------------------

trainer.train()