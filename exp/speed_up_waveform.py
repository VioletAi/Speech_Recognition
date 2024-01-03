import torchaudio
import torch
import os
import json

# Function to apply speed perturbation to a waveform
def apply_speed_perturbation(waveform, sample_rate, speed_factor):
    # Changing speed by resampling
    new_sample_rate = int(sample_rate * speed_factor)
    resampled_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)
    return resampled_waveform, new_sample_rate

# Function to convert waveform to FBank features with speed perturbation
def waveform_to_fbank_with_perturbation(waveform_path, speed_factor):
    waveform, sample_rate = torchaudio.load(waveform_path)
    # Apply speed perturbation
    perturbed_waveform, new_sample_rate = apply_speed_perturbation(waveform, sample_rate, speed_factor)
    # Convert to FBank features
    fbank_features = torchaudio.compliance.kaldi.fbank(perturbed_waveform, num_mel_bins=23, sample_frequency=new_sample_rate)
    return fbank_features

# Function to save FBank features and create JSON entries
def process_and_save(data_json_path, fbank_save_dir, output_json_path):
    with open(data_json_path, 'r') as f:
        data_info = json.load(f)

    output_json_info = {}

    for item_id, item_info in data_info.items():
        waveform_path = item_info['wav']
        speaker_id = item_info['spk_id']

        # Process original, slowed, and sped-up versions
        for speed_factor in [1.0, 0.9, 1.1]:  # Original, slow down, speed up
            # Convert waveform to FBank features with speed perturbation
            fbank_features = waveform_to_fbank_with_perturbation(waveform_path, speed_factor)

            # Extract wave_id
            wav_id = (item_id.split('_')[1]).split('.')[0]+"_speed"+str(speed_factor)

            # Save FBank features
            fbank_file_path = os.path.join(fbank_save_dir, speaker_id, wav_id)

            #create a directory recursively
            os.makedirs(os.path.dirname(fbank_file_path), exist_ok=True)

            torch.save(fbank_features, fbank_file_path)

            #Create item id
            item_id_fbank = speaker_id + '_' + wav_id

            # Create entry for output JSON
            output_json_info[item_id_fbank] = {
                'fbank': fbank_file_path,
                'spk_id': speaker_id,
                'duration': float(item_info['duration'])*speed_factor,
                'phn': item_info['phn']
            }

    # Save output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(output_json_info, f, indent=4)


# Paths to your original JSON files
meta_train_json_path = './train.json'
meta_dev_json_path = './dev.json'
meta_test_json_path = './test.json'

# Directory to save FBank features

fbank_save_dir_train = '/rds/user/wa285/hpc-work/MLMI2/exp/fbank_data_agumentation/train/'
#fbank_save_dir_dev = '/rds/user/wa285/hpc-work/MLMI2/exp/fbank_data_agumentation/dev/'
#fbank_save_dir_test = '/rds/user/wa285/hpc-work/MLMI2/exp/fbank_data_agumentation/test/'

# Process and save for each dataset
process_and_save(meta_train_json_path, fbank_save_dir_train, 'train_fbank_augmentation.json')
#process_and_save(meta_dev_json_path, fbank_save_dir_dev, 'dev_fbank_augmentation.json')
#process_and_save(meta_test_json_path, fbank_save_dir_test, 'test_fbank_augmentation.json')

print("Augmentated Data Processing Complete!")