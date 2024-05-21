data_path = "human_chat.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().lower().split('\n')

# group lines by pairs

pairs = [(lines[i], lines[i+1])  for i in range(len(lines)-1) if "human 1: hi" not in lines[i+1]]
      