from jiwer import compute_measures, cer
import torch
import Levenshtein

from dataloader import get_dataloader
from utils import concat_inputs
import numpy as np


def decode(model, args, json_file, char=False):
    with open("phone_map", 'r') as file:
        phone_map_contents = file.read()
    phone_map_lines = phone_map_contents.split('\n')
    phone_map_dict = {line.split(': ')[0]: line.split(': ')[1] for line in phone_map_lines if line}

    idx2grapheme = {y: x for x, y in args.vocab.items()}
    test_loader = get_dataloader(json_file, 1, False)
    stats = [0., 0., 0., 0.]

    for data in test_loader:
        inputs, in_lens, trans, _ = data
        inputs = inputs.to(args.device)
        in_lens = in_lens.to(args.device)
        inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
        with torch.no_grad():
            outputs = torch.nn.functional.softmax(model(inputs), dim=-1)
            outputs = torch.argmax(outputs, dim=-1).transpose(0, 1)
        outputs = [[idx2grapheme[i] for i in j] for j in outputs.tolist()]
        
        outputs = [[v for i, v in enumerate(j) if i == 0 or v != j[i - 1]] for j in outputs]
        outputs = [list(filter(lambda elem: elem != "_", i)) for i in outputs]
        
        #print(outputs)
        
        mapped_output_phones_39 = [[phone_map_dict[phone] for phone in outputs[0] if phone in phone_map_dict]]
#         mapped_output_phones_39=mapped_output_phones_39.to(args.device)
        
        #print(mapped_output_phones_39)
        
        outputs = [" ".join(i) for i in mapped_output_phones_39]
        
        if char:
            cur_stats = cer(trans, outputs, return_dict=True)
        else:
            cur_stats = compute_measures(trans, outputs)

        stats[0] += cur_stats["substitutions"]
        stats[1] += cur_stats["deletions"]
        stats[2] += cur_stats["insertions"]
        stats[3] += cur_stats["hits"]

    total_words = stats[0] + stats[1] + stats[3]
    sub = stats[0] / total_words * 100
    dele = stats[1] / total_words * 100
    ins = stats[2] / total_words * 100
    cor = stats[3] / total_words * 100
    err = (stats[0] + stats[1] + stats[2]) / total_words * 100
    
    #f.close()
    return sub, dele, ins, cor, err
