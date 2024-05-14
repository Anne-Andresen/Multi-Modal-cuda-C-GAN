
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

class CrossAttention : public torch::nn::Module {
public:
    CrossAttention(int64_t embed_dim, int64_t num_heads) {
        attention = register_module("attention", torch::nn::MultiheadAttention(embed_dim, num_heads));
    }

    torch::Tensor forward(torch::Tensor queries, torch::Tensor key, torch::Tensor values) {
        int64_t batch_size = queries.size(0);
        int64_t depth = queries.size(1);
        int64_t height = queries.size(2);
        int64_t width = queries.size(3);

        queries = queries.permute({1, 0, 2, 3}).reshape({depth, -1, queries.size(-1)});
        key = key.permute({1, 0, 2, 3}).reshape({depth, -1, key.size(-1)});
        values = values.permute({1, 0, 2, 3}).reshape({depth, -1, values.size(-1)});

        torch::Tensor output;
        AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "CrossAttention forward", ([&] {
            output = attention->forward(queries, key, values).toTuple()->get<0>();
        }));

        output = output.view({depth, batch_size, -1, queries.size(-1), queries.size(-1)});
        output = output.permute({1, 2, 0, 3, 4});

        return output;
    }

private:
    torch::nn::MultiheadAttention attention;
};
