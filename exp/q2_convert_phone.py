def create_vocab_and_mapping(file_path):
    # Read the phone mapping from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Create a dictionary for the mapping
    phone_mapping = {}
    for line in lines:
        original, reduced = line.strip().split(':')
        phone_mapping[original.strip()] = reduced.strip()

    # Create a reduced set of phones
    reduced_phones = list(set(phone_mapping.values()))[1:]
    reduced_phones = ['_'] + reduced_phones

    # Write to vocab_39.txt
    with open('vocab_39.txt', 'w') as vocab_file:
        #Add blank token for CTC
        for phone in reduced_phones:
            vocab_file.write(phone +'\n')

    print("Vocab file created with", len(reduced_phones), "phones.")

# Replace 'your_phone_map_file_path' with the path to your phone_map.txt file
#create_vocab_and_mapping('/rds/user/wa285/hpc-work/MLMI2/exp/phone_map')

# Visualise phone freqency
import json
import matplotlib.pyplot as plt
from collections import Counter

# Path to the uploaded train.json file
train_json_path = '/rds/user/wa285/hpc-work/MLMI2/exp/train.json'

# Reading the train.json file
with open(train_json_path, 'r') as file:
    train_data_full = json.load(file)

# Counting the frequency of each phone in the full dataset
phone_counter_full = Counter()
for item in train_data_full.values():
    phones = item['phn'].split()
    phone_counter_full.update(phones)

# Plotting the distribution for the full dataset
plt.figure(figsize=(15, 8))
plt.bar(phone_counter_full.keys(), phone_counter_full.values())
plt.xlabel('Phone')
plt.ylabel('Frequency')
plt.title('Distribution of Phone Frequencies in the Full Training Set')
plt.xticks(rotation=90)

# Save the figure as a PNG file
png_file_path = '/rds/user/wa285/hpc-work/MLMI2/exp/phone_distribution.png'
plt.savefig(png_file_path, bbox_inches='tight')

# Informing the user about the saved file
print(f"The plot has been saved as a PNG file at: {png_file_path}")





