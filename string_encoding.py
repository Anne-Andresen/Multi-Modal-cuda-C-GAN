import torch

def encoding(text, dose):
    collective_string = text + ' ' + dose
    vocabulary = 'abcdefghijklmnopqrstuvwxyz,._ -1234567890'
    chars = sorted(list(set(collective_string)))
    vocab = sorted(list(set(vocabulary)))
    size = len(vocab)
    vocab_size = len(chars)
    #print(''.join(chars))
    #print('vocab size: ', size)
    mapping = {ch:i for i,ch in enumerate(vocab) }
    encoded = lambda s: [mapping[c] for c in s]
    #final_Encoding = encoded(collective_string)
    #print('encoded: ', final_Encoding)
    data = torch.tensor(encoded(collective_string), dtype=torch.float32)
    data = data/10
    repeat_factor = 2097152 // data.numel() 
    repeat_tensor = data.repeat(repeat_factor+1)
    repeat_tensor = repeat_tensor[:-1]
    dat = torch.reshape(repeat_tensor, (1, 1, 128, 128, 128))
    #dat = dat.unsqueeze(0).unsqueeze(0)
    return dat




