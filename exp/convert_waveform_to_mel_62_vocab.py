import torchaudio
import torch
import os
import json

# Function to convert waveform to FBank features
def waveform_to_fbank(waveform_path):
    waveform, sample_rate = torchaudio.load(waveform_path)
    fbank_features = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=23, sample_frequency=sample_rate)
    return fbank_features

# Function to save FBank features and create JSON entries
def process_and_save(data_json_path, fbank_save_dir, output_json_path):
    with open(data_json_path, 'r') as f:
        data_info = json.load(f)

    output_json_info = {}

    for item_id, item_info in data_info.items():
        waveform_path = item_info['wav']
        speaker_id = item_info['spk_id']

        # Convert waveform to FBank features
        #fbank_features = waveform_to_fbank(waveform_path)

        # Extract wave_id
        wav_id = (item_id.split('_')[1]).split('.')[0]

        # Save FBank features
        fbank_file_path = os.path.join(fbank_save_dir, speaker_id, wav_id)
        
        #create a directory recursively
        #os.makedirs(os.path.dirname(fbank_file_path), exist_ok=True)

        #Save the fbank feature to the path
        #torch.save(fbank_features, fbank_file_path)

        #Create item id
        item_id_fbank = speaker_id + '_' + wav_id


        # Create entry for output JSON
        output_json_info[item_id_fbank] = {
            'fbank': fbank_file_path,
            'spk_id': speaker_id,
            'duration': item_info['duration'],
            'phn': item_info['phn']
        }

    # Save output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(output_json_info, f, indent=4)

# Paths to your original JSON files
meta_train_json_path = './train_p61.json'
# meta_dev_json_path = './dev.json'
# meta_test_json_path = './test.json'

# Directory to save FBank features

fbank_save_dir_train = '/rds/user/wa285/hpc-work/MLMI2/exp/fbank/train/'
# fbank_save_dir_dev = '/rds/user/wa285/hpc-work/MLMI2/exp/fbank/dev/'
# fbank_save_dir_test = '/rds/user/wa285/hpc-work/MLMI2/exp/fbank/test/'

# Process and save for each dataset
process_and_save(meta_train_json_path, fbank_save_dir_train, 'train_fbank_p61.json')
# process_and_save(meta_dev_json_path, fbank_save_dir_dev, 'dev_fbank.json')
# process_and_save(meta_test_json_path, fbank_save_dir_test, 'test_fbank.json')

print("Processing complete!")
