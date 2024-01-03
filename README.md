# TIMIT Speech Recognition Using CTC

## Introduction
This repository contains the implementation of an end-to-end Automatic Speech Recognition (ASR) system using the Connectionist Temporal Classification (CTC) loss applied to the TIMIT acoustic-phonetic corpus.

## Structure
- `exp/`: Contains code for training and decoding and Jupyter Notebooks for analysis and visualisation.
- `report.pdf`: The detailed report of the project.

## Pre-requisites
- Python 3.7+
- PyTorch 2.1.0
- Required Python packages are listed in `requirements.txt`.

## Installation
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.

## Usage
Before running the code, ensure you download [train_json.zip](https://drive.google.com/file/d/1a-c48PQIbUO8BSRVzKnOmgSe5-yzx9xU/view?usp=drive_link), [checkpoints.zip](https://drive.google.com/file/d/1M74cCdRV3V7o1kIzdxTGDctlCpt5G-5C/view?usp=drive_link) and [fbank.zip](https://drive.google.com/file/d/1M74cCdRV3V7o1kIzdxTGDctlCpt5G-5C/view?usp=drive_link). Decompress them in the `exp/` folder. Make sure the code follows the following structure:
```
├── project/
│   ├── exp/
│   │   ├── checkpoints/
│   │   │   ├── folder named after timestamp
│   │   ├── fbank/
│   │   │   ├── folder named after speaker_id
│   │   ├── train.json
│   │   ├── train_fbank.json
│   │   ├── (other json files unzip from train_json.zip)
│   │   └── (other code and resources)
│   ├── .git/
│   └── report.pdf
```

1. Activate the provided Anaconda environment.
2. Navigate to `exp/` and run the desired scripts. exp/MLMI2_Lab.ipynb contains the majority of the code.

## Data
The TIMIT dataset is used, which must be processed into log Mel filterbanks (FBank) or Mel-frequency cepstral coefficients (MFCCs) before use.

## Reporting Issues
If you encounter any issues, please open an issue on the GitHub repository page with a detailed description of the problem.

## Authors
- Weijia Ai

