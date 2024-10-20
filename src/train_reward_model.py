
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from prepare_dataset import prepare_dataset

def train_reward_model():
    parsed_data_with_feedback = prepare_dataset()
    
    reward_dataset = Dataset.from_dict({
        'input': [f"Human: {item['query']} Assistant: {item['response']}" for item in parsed_data_with_feedback],
        'reward': [item['feedback'] for item in parsed_data_with_feedback]
    })
    
    train_test_split = reward_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples['input'], truncation=True, padding=True, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["input"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["input"])
    
    train_dataset = train_dataset.rename_column("reward", "labels")
    eval_dataset = eval_dataset.rename_column("reward", "labels")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = GPT2ForSequenceClassification.from_pretrained("gpt2-medium", num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    training_args = TrainingArguments(
        output_dir="./data/trained_reward_model",
        evaluation_strategy="steps",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10000,
        save_steps=10000,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    model.save_pretrained("./data/trained_reward_model")
    tokenizer.save_pretrained("./data/trained_reward_model")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_reward_model()
