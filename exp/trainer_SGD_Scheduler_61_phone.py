from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from torch.optim import SGD
from phone_61_decoder import decode
from utils import concat_inputs
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import get_dataloader

def train(model, args):
    
        # Parse the phone_map file
    with open("phone_map", 'r') as file:
        phone_map_contents = file.read()
    phone_map_lines = phone_map_contents.split('\n')
    phone_map_dict = {line.split(': ')[0]: line.split(': ')[1] for line in phone_map_lines if line}

    # Your 62-phone set
    phone_set_62 = args.vocab # Complete this with all 62 phones

    # Assuming you have a similar dictionary for the 39-phone set
    phone_set_39 = args.vocab_39
    

    # Create index mapping
    index_mapping = {phone_set_62[src]: phone_set_39[dst] for src, dst in phone_map_dict.items() if src in phone_set_62 and dst in phone_set_39}
    
    print()
    print(phone_set_62)
    print()
    print(phone_set_39)
    
    print()
    index_mapping[0]=0
    
    print(index_mapping)
    
    #print(index_mapping)
 
    torch.manual_seed(args.seed)
    train_loader = get_dataloader(args.train_json, args.batch_size, True)
    val_loader = get_dataloader(args.val_json, args.batch_size, False)
    criterion = CTCLoss(zero_infinity=True)
    optimiser = SGD(model.parameters(), lr=args.lr)

    # Define the LR scheduler
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=0, verbose=True)

    def train_one_epoch(epoch):
        running_loss = 0.
        last_loss = 0.

        for idx, data in enumerate(train_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)

            optimiser.zero_grad()
            outputs = log_softmax(model(inputs), dim=-1)           
            
            loss = criterion(outputs, targets, in_lens, out_lens)
            loss.backward()
            
            #add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()

            running_loss += loss.item()
            if idx % args.report_interval + 1 == args.report_interval:
                last_loss = running_loss / args.report_interval
                print('  batch {} loss: {}'.format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.
        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path('checkpoints/{}'.format(timestamp)).mkdir(parents=True, exist_ok=True)
    best_val_loss = 100

    for epoch in range(args.num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        running_val_loss = 0.
        for idx, data in enumerate(val_loader):
            inputs, in_lens, trans, _ = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            targets = [torch.tensor(list(map(lambda x: args.vocab_39[x], target.split())),
                                    dtype=torch.long)
                       for target in trans]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long)
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)
            
            outputs = model(inputs)
            
                        #print(outputs.size())
            
            mapped_output = torch.zeros(outputs.shape[0], outputs.shape[1], len(phone_set_39))
            mapped_output = mapped_output.to(args.device)

            # Map the output
            
            for src_idx, dst_idx in index_mapping.items():
                mapped_output[:, :, dst_idx] += outputs[:, :, src_idx] 
            
            #print(mapped_output.size())
            mapped_output = log_softmax(mapped_output, dim=-1)
            #add the corresponding outputs based on mapping
            
            val_loss = criterion(mapped_output, targets, in_lens, out_lens)
            running_val_loss += val_loss
        avg_val_loss = running_val_loss / len(val_loader)
        val_decode = decode(model, args, args.val_json)


        # Step the scheduler
        scheduler.step(val_decode[4])


        print('LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%'.format(
            avg_train_loss, avg_val_loss, val_decode[4])
            )
        


        if val_decode[4] < best_val_loss:
            best_val_loss = val_decode[4]
            model_path = 'checkpoints/{}/model_{}'.format(timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
    return model_path
