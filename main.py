import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load your model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

correct = 0  # Number of correct predictions
total = 0  # Total number of predictions

# Open the HellaSwag jsonl file
with open('hellaswag/data/hellaswag_val.jsonl', 'r') as f:
    # Loop through each line in the file
    for line in f:
        item = json.loads(line)  # Parse each line as a JSON object

        context = item['ctx']  # Use 'ctx' field as context
        endings = item['endings']  # Possible endings
        correct_end = item['label']  # Correct answer index is in 'label' field

        # Generate a score for each possible ending
        scores = []
        for ending in endings:
            input_text = context + ' ' + ending  # Combine context and ending
            input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)  # Get model output
            loss = outputs.loss  # Get loss value
            scores.append(loss.item())  # Add loss value to scores list

        # Choose the most likely ending based on the scores
        predicted_end = scores.index(min(scores))

        # Update the number of correct predictions
        if predicted_end == correct_end:
            correct += 1
        total += 1  # Update the total number of predictions

        # Print the accuracy
        print("Accuracy({}/{}): {}".format(correct, total, correct / total), end='\r')
