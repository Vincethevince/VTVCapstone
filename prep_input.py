data_path = "human_chat.txt"
data_path = "dialogs.txt"

# Defining lines as a list of each line
try:
  with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().lower().split('\n')
except FileNotFoundError:
    print(f"Error: File {data_path} not found.")
    lines = []

# Ensure there are enough lines to form pairs
if len(lines) < 2:
    print("Not enough lines to form pairs.")
    pairs = []
else:
  # group lines by pairs
  #pairs = [(lines[i], lines[i+1])  for i in range(len(lines)-1) if "human 1: hi" not in lines[i+1]]
  pairs = [[text for text in line.strip().split("\t")] for line in lines]
  print(pairs)
      
