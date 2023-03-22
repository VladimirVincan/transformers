import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate

from transformers import WhisperForConditionalGeneration


from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset



AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"



def normalize_dataset(ds, audio_column_name=None, text_column_name=None):
    if audio_column_name is not None and audio_column_name != AUDIO_COLUMN_NAME:
        ds = ds.rename_column(audio_column_name, AUDIO_COLUMN_NAME)
    if text_column_name is not None and text_column_name != TEXT_COLUMN_NAME:
        ds = ds.rename_column(text_column_name, TEXT_COLUMN_NAME)
    # resample to the same sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    # normalise columns to ["audio", "sentence"]
    ds = ds.remove_columns(set(ds.features.keys()) - set([AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME]))
    return ds

import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"]#.to(torch.half).contiguous()

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


common_voice_datasets = DatasetDict()
common_voice_datasets = load_dataset("audiofolder", data_dir=os.path.join("cv-corpus-12.0-2022-12-07", "sr"))

common_voice_datasets["train"] = normalize_dataset(common_voice_datasets["train"])
common_voice_datasets["test"] = normalize_dataset(common_voice_datasets["test"])
common_voice_datasets["validation"] = normalize_dataset(common_voice_datasets["validation"])

common_voice_datasets["train"] = concatenate_datasets([common_voice_datasets["train"], common_voice_datasets["validation"]])

#max_input_length = 30
#min_input_length = 0
#def is_audio_in_length_range(length):
#    return length > min_input_length and length < max_input_length

#common_voice_datasets = common_voice_datasets.filter(
#    is_audio_in_length_range, num_proc=6, input_columns=["audio"]
#)
'''
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    PolarityInversion,
    TimeStretch,
)

musan_dir = "./musan"

# define augmentation
augmentation = Compose(
    [
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        OneOf(
            [
                AddBackgroundNoise(
                    sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0, noise_transform=PolarityInversion(), p=1.0
                ),
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
            ],
            p=0.2,
        ),
    ]
)


def augment_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    sample = batch[AUDIO_COLUMN_NAME]

    # apply augmentation
    augmented_waveform = augmentation(sample["array"], sample_rate=sample["sampling_rate"])
    batch[AUDIO_COLUMN_NAME]["array"] = augmented_waveform
    return batch


preprocessing_num_workers = 4

augmented_raw_training_dataset = common_voice_datasets["train"].map(
    augment_dataset, num_proc=preprocessing_num_workers, desc="augment train dataset"
)

# combine
common_voice_datasets["train"] = concatenate_datasets([common_voice_datasets["train"], augmented_raw_training_dataset])
common_voice_datasets["train"] = common_voice_datasets["train"].shuffle(seed=10)
'''

model_size="tiny"
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-" + model_size, language="serbian", task="transcribe")
#from transformers import AutoProcessor
#processor = AutoProcessor.from_pretrained("./finetuned-whisper-RTS-tiny")


print("------------- Spectrogram creation started -------------")
common_voice_datasets = common_voice_datasets.map(prepare_dataset, remove_columns=next(iter(common_voice_datasets.values())).column_names, num_proc=6)
print("------------- Spectrogram creation finished -------------")


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

#model = WhisperForConditionalGeneration.from_pretrained("./finetuned-whisper-commonvoice-tiny/checkpoint-40000")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-" + model_size)

#model.freeze_encoder()

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache=False, # todo moze se postaviti na True ako se gradient_checkpointing ispod postavi na False, bice brzi trening, ali vece zauzece memorije


useF16 = True
if DEVICE == "cpu":
    useF16 = False
useF16 = False

from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-whisper-commonvoice-" + model_size,  # change to a repo name of your choice
    per_device_train_batch_size=4, #default 16
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size - default 1
    learning_rate=1e-7,
    warmup_steps=500,
    max_steps=40000,
    gradient_checkpointing=True,
    fp16=useF16,
    fp16_full_eval=useF16,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice_datasets["train"],
    eval_dataset=common_voice_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

train_result = trainer.train()
model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

eval_metrics = trainer.evaluate(metric_key_prefix="eval")

print(eval_metrics)

print("------------------")
'''

# Generate predictions for the test samples
predictions = trainer.predict(common_voice_datasets["test"])

# Loop through the test samples and print the labels and predictions
for i, sample in enumerate(common_voice_datasets["test"]):

    print(f"  Label: {processor.tokenizer.decode(sample['labels'], skip_special_tokens=True)}")
    print(f"  Prediction: {processor.tokenizer.decode(predictions[0][i], skip_special_tokens=True)}")
    print('------------------------------------------')
    
'''
