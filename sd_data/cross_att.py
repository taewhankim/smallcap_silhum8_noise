import torch
import torch.utils.checkpoint
from torch import nn, einsum
from transformers.pytorch_utils import Conv1D
from typing import Optional, Tuple, Union
from sd_data.config import CA_config
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from sd_data.util import ACT2FN

#################################################################

class CrossAttention(nn.Module):
    def __init__(self, embed_dim,  split_size =768, cross_attention_reduce_factor=4, is_cross_attention=True, num_heads = 12, head_dim=64, dropout=0.1, config= None):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.is_cross_attention = is_cross_attention
        self.resid_dropout = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor

        max_positions = 1024
        self.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions)

        if self.is_cross_attention:
            self.c_attn = Conv1D(int(2 / self.cross_attention_reduce_factor * self.embed_dim),
                                 ###(768,384) (2/4)인 이유는 k,v 2개 이기때문
                                 self.embed_dim)
            self.q_attn = Conv1D(int(self.embed_dim / self.cross_attention_reduce_factor), self.embed_dim)  ###(768,192)
            self.c_proj = Conv1D(self.embed_dim, int(self.embed_dim / self.cross_attention_reduce_factor))  ###(192,768)
        else:
            self.c_attn = Conv1D(3 * int(self.embed_dim / self.cross_attention_reduce_factor), self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, int(self.embed_dim / self.cross_attention_reduce_factor))

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / torch.tensor(
            value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )
        # if not self.is_cross_attention:
        #     # if not self.is_cross_attention:
        #     # if only "normal" attention layer implements causal mask
        #     query_length, key_length = query.size(-2), key.size(-2)
        #     causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].to(torch.bool)
        #     mask_value = torch.finfo(attn_weights.dtype).min
        #     # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        #     # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        #     mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        #     attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)


    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        split_size = int(self.split_size / self.cross_attention_reduce_factor)  ## 192
        head_dim = int(self.head_dim / self.cross_attention_reduce_factor)  ## 16

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            # tmp = np.zeros(encoder_hidden_states.shape[0], encoder_hidden_states[-1]).astype('uint8')
            # encoder_attention_mask = torch.tensor(tmp, dtype=torch.bool, requires_grad=False, device=encoder_hidden_states.device)
            ## query: text # key, value: image
            query = self.q_attn(hidden_states)  ## (B,140,768) -->(B,140,192)
            key, value = self.c_attn(encoder_hidden_states).split(split_size, dim=2)  ##(B,50,768) --> ( B,50,192)
            attention_mask = encoder_attention_mask

            query = self._split_heads(query, self.num_heads, head_dim)
            key = self._split_heads(key, self.num_heads, head_dim)
            value = self._split_heads(value, self.num_heads, head_dim)
        else:
            query, key, value = self.c_attn(hidden_states).split(split_size, dim=2)

            query = self._split_heads(query, self.num_heads, head_dim)
            key = self._split_heads(key, self.num_heads, head_dim)
            value = self._split_heads(value, self.num_heads, head_dim)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads,
                                        int(self.head_dim / self.cross_attention_reduce_factor))
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = attn_output
        # if output_attentions:
        #     outputs += (attn_weights)

        return outputs

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.,output_features = None):
        super().__init__()
        if output_features is None:
            out_features = in_features
        else:
            out_features = output_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):

    def __init__(self, embed_dim= 768, num_heads = 12, head_dim = 64, drop=0.1, mlp_ratio = 4, norm_layer=nn.LayerNorm, act_layer=nn.GELU, config=None):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.norm3 = norm_layer(embed_dim)

        self.attn = CrossAttention(embed_dim, split_size =config['split_size'], cross_attention_reduce_factor=config["cross_attention_reduce_factor"],
                                   is_cross_attention=config['is_cross_attention'], num_heads=num_heads, head_dim=head_dim, dropout=drop, config= config)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim*mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, query_embed, kv_embed):
        residual = query_embed
        # cross_attn_outputs = self.attn(hidden_states = self.norm1(query_embed), encoder_hidden_states = self.norm2(kv_embed))

        norm_query_embed = self.norm1(query_embed)
        if kv_embed is not None:
            norm_kv_embed = self.norm2(kv_embed)
        else:
            norm_kv_embed = kv_embed
        cross_attn_outputs = self.attn(hidden_states = norm_query_embed, encoder_hidden_states = norm_kv_embed)

        query_embed = cross_attn_outputs + residual ## residual

        query_embed = query_embed + self.mlp(self.norm3(query_embed)) ## norm->MLP->res
        return query_embed


class CROSS_ATTENTION(nn.Module):
    def __init__(self, pre_config=None):
        super().__init__()
        if pre_config is None:
            config = CA_config
        else:
            config = pre_config
        embed_dim = config["embed_dim"]
        blocks_num = config["blocks_num"]
        norm_layer = nn.LayerNorm
        activation_function = "gelu_new"
        drop_p = config["drop_p"]
        head_dim = config["head_dim"]
        num_heads = config['num_heads']
        mlp_ratio = config["mlp_ratio"]
        act_layer = ACT2FN[activation_function]# nn.GELU
        self.config = config
        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads, head_dim = head_dim, drop=drop_p, mlp_ratio = mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, config=config)
            for _ in range(blocks_num)])

        # final norm
        self.norm = norm_layer(embed_dim)

    # def forward(self, x):
    #     x = self.blocks(x)
    #     x = self.norm(x)
    #     return x
    def forward(self, hidden_states = None, encoder_hidden_states = None):
        for block in self.blocks:
            hidden_states = block(hidden_states,encoder_hidden_states)
        hidden_states = self.norm(hidden_states)

        return hidden_states
