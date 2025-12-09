import re
from ast import literal_eval

def parse_training_log(log_content):
    """
    Parses the training log content to extract loss and rewards/margins.

    Args:
        log_content (str): The entire content of the training log file.

    Returns:
        tuple: A tuple containing two lists: (losses, margins).
    """
    losses = []
    margins = []

    # Regex to find all dictionary-like strings in the log content
    # It looks for a '{' followed by any characters (non-greedy, including newlines) 
    # until a closing '}'. The re.DOTALL flag is crucial for matching across newlines.
    dict_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    
    # Find all matches
    dict_strings = dict_pattern.findall(log_content)
    
    for dict_str in dict_strings:
        try:
            # Safely evaluate the string as a Python dictionary
            data = literal_eval(dict_str)
            
            # Extract 'loss' if it exists
            if 'loss' in data:
                losses.append(data['loss'])
            
            # Extract 'rewards/margins' if it exists
            if 'rewards/margins' in data:
                margins.append(data['rewards/margins'])
                
        except (ValueError, SyntaxError) as e:
            # Skip any malformed or incomplete dictionary strings
            # print(f"Skipping malformed entry: {dict_str} - Error: {e}")
            continue

    return losses, margins

# --- Example Usage ---

# 1. Define the log content (replace this with file reading in a real scenario)
def read_log_file(file_path):
     with open(file_path, 'r') as f:
         return f.read()

log_content = read_log_file("train.log")

# 2. Get the lists
extracted_losses, extracted_margins = parse_training_log(log_content)

# 3. Print the results
print("## ✅ Extracted Values")
print(f"Total metrics entries found: {len(extracted_losses)}")
#print("\n**Loss Values (First 5):**")
#print(extracted_losses[:5])
#print("\n**Rewards/Margins Values (First 5):**")
#print(extracted_margins[:5])


print("\n**Loss Values**")
print(extracted_losses)
print("\n**Rewards/Margins Values **")
print(extracted_margins)


# --- How to read from a file: ---
# extracted_losses, extracted_margins = parse_training_log(file_content)
# ...
