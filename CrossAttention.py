class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        #self.tensor1 = tensor1
        #self.tensor2 = tensor2


    def forward(self, queries, key, values):
        batch_size, _, depth, heaight, width = queries.size()
        queries = queries.permute(2, 0, 1, 3, 4).reshape(depth, -1, queries.size(-1))
        key = key.permute(2, 0, 1, 3, 4).reshape(depth, -1, key.size(-1))
        values = values.permute(2, 0, 1, 3, 4).reshape(depth, -1, values.size(-1))
        output, _ = self.attention(queries, key, values)

        output = output.view(depth, batch_size, -1, queries.size(-1), queries.size(-1))
        #print('output shape: ', output.shape)
        output = output.permute(1, 2, 0, 3, 4)
        print('output shape: ', output.shape)
        return output
