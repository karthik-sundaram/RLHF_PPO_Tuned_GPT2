
from datasets import load_dataset

def prepare_dataset():
    dataset = load_dataset("yitingxie/rlhf-reward-datasets", split="train")
    
    parsed_data_with_feedback = []

    for item in dataset:
        if 'prompt' in item and 'chosen' in item and 'rejected' in item:
            query = item['prompt']
            response_chosen = item['chosen']
            response_rejected = item['rejected']
            
            
            if len(query) < 500 and len(response_chosen) < 500 and len(response_rejected) < 500:
            
                parsed_data_with_feedback.append({
                    'query': query,
                    'response': response_chosen,
                    'feedback': 1.0 
                })
                parsed_data_with_feedback.append({
                    'query': query,
                    'response': response_rejected,
                    'feedback': -1.0  
                })
    
    return parsed_data_with_feedback

if __name__ == "__main__":
    data = prepare_dataset()
    for entry in data[:5]:
        print(f"Query: {entry['query']}")
        print(f"Response: {entry['response']}")
        print(f"Feedback: {entry['feedback']}")
        print("=" * 50)
