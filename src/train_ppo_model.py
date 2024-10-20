# src/train_ppo_model.py

import random
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
import torch
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils
from prepare_dataset import prepare_dataset

def train_ppo_model():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    
    reward_model = GPT2ForSequenceClassification.from_pretrained("./data/trained_reward_model")
    reward_tokenizer = GPT2Tokenizer.from_pretrained("./data/trained_reward_model")
    
    config = PPOConfig(
        model_name="gpt2-medium",
        learning_rate=1e-7,
        batch_size=4,
        mini_batch_size=4,
        cliprange=0.02,
        target_kl=0.02,
    )
    
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2-medium")
    ppo_model.pretrained_model.resize_token_embeddings(len(tokenizer))
    
    optimizer = Adam(ppo_model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.95)
    
    ppo_trainer = PPOTrainer(
        model=ppo_model,
        config=config,
        tokenizer=tokenizer,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_model.to(device)
    
    total_steps = 500
    batch_size = config.batch_size
    
    parsed_data_with_feedback = prepare_dataset()
    
    def parse_numerical_stats(stats):
        for key, value in stats.items():
            if isinstance(value, (int, float)) and key == 'objective/kl':
                print(f"{key}: {value}")
                
    def compute_rewards(queries, responses):
        rewards = []
        for query, response in zip(queries, responses):
            input_text = f"Human: {query} Assistant: {response}"
            inputs = reward_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                output = reward_model(**inputs)
                reward_score = output.logits.item()
            rewards.append(reward_score)
        return rewards
    
    def prepare_batch(data, tokenizer, batch_size, device):
        random.shuffle(data)
        batch_data = data[:batch_size]
        queries = [item['query'] for item in batch_data]
        responses = [item['response'] for item in batch_data]
        feedbacks = [item['feedback'] for item in batch_data]
        
        query_tensors = [torch.tensor(tokenizer.encode(q, truncation=True)).to(device) for q in queries]
        response_tensors = [torch.tensor(tokenizer.encode(r, truncation=True)).to(device) for r in responses]
        scores = [torch.tensor(score, dtype=torch.float32).to(device) for score in feedbacks]
        
        return query_tensors, response_tensors, scores
    
    with torch.autograd.detect_anomaly():
        for step in tqdm(range(1, total_steps + 1)):
           
            query_tensors, response_tensors, scores = prepare_batch(parsed_data_with_feedback, tokenizer, batch_size, device)
            
         
            for name, tensors in zip(["Queries", "Responses"], [query_tensors, response_tensors]):
                for idx, tensor in enumerate(tensors):
                    if torch.any(tensor >= ppo_model.config.vocab_size):
                        raise ValueError(f"{name} at index {idx} contains token IDs >= vocab size.")
            
            try:
             
                stats = ppo_trainer.step(
                    queries=query_tensors,
                    responses=response_tensors,
                    scores=scores
                )
    
            
                kl_divergence = stats.get('objective/kl', 0)
                if kl_divergence < -2 or kl_divergence > 2:
                    print(f"Stopping early at step {step}: KL divergence is out of bounds ({kl_divergence})")
                    break
    
                torch.cuda.synchronize()
    
            except RuntimeError as e:
                print(f"RuntimeError at step {step}: {e}")
                break
    
          
            torch.nn.utils.clip_grad_norm_(ppo_model.parameters(), max_norm=1.0)
    
         
            scheduler.step()
    
       
            torch.cuda.empty_cache()
            parse_numerical_stats(stats)
            
 
    ppo_model.to("cpu")
    ppo_model.save_pretrained("./data/ppo_gpt2_model")
    tokenizer.save_pretrained("./data/ppo_gpt2_tokenizer")

if __name__ == "__main__":
    train_ppo_model()
