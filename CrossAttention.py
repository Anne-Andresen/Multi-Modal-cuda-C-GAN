import SimpleITK as sitk
import numpy as np
import glob
import torch
import nibabel as nib
import torch.nn as nn
import torch



class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)


    def forward(self, queries, key, values):
        batch_size, _, depth, heaight, width = queries.size()
        queries = queries.permute(2, 0, 1, 3, 4).reshape(depth, -1, queries.size(-1))
        key = key.permute(2, 0, 1, 3, 4).reshape(depth, -1, key.size(-1))
        values = values.permute(2, 0, 1, 3, 4).reshape(depth, -1, values.size(-1))
        output, _ = self.attention(queries, key, values)

        output = output.view(depth, batch_size, -1, queries.size(-1), queries.size(-1))
       
        output = output.permute(1, 2, 0, 3, 4)
        print('output shape: ', output.shape)
        return output





image_path = '/1.nii.gz'
struct_path = '/11.nii.gz'
image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
struct = sitk.GetArrayFromImage(sitk.ReadImage(struct_path))
torch_tensor_img = torch.from_numpy(image).to(torch.float32).unsqueeze(0).unsqueeze(0)
torch_tensor_struct = torch.from_numpy(struct).to(torch.float32).unsqueeze(0).unsqueeze(0)

embed_dim = 128
num_heads = 8
cross_attention = CrossAttention(embed_dim, num_heads)
output = cross_attention(torch_tensor_img, torch_tensor_struct, torch_tensor_struct)

if __name__ == "main":
    ''
