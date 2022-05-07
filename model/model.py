import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.cuda.amp import autocast as autocast

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU()

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RWSA(nn.Module):
    def __init__(self, __C, hidesize, hidesizehead, mulhead, midsize):
        super(RWSA, self).__init__()
        self.__C = __C
        self.hidesize = hidesize
        self.hidesizehead = hidesizehead
        self.mulhead = mulhead
        self.midsize = midsize

        self.linear_v = nn.Linear(hidesize, hidesize)
        self.linear_k = nn.Linear(hidesize, midsize*mulhead)
        self.linear_q = nn.Linear(hidesize, midsize*mulhead)
        self.linear_merge = nn.Linear(hidesize, hidesize)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.mulhead,
            self.hidesizehead
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.mulhead,
            self.midsize
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.mulhead,
            self.midsize
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidesize
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -5e4)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, __C, hidesize, ffsize):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidesize,
            mid_size=ffsize,
            out_size=hidesize,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, __C, hidesize, hidesizehead, mulhead, ffsize):
        super(SA, self).__init__()

        self.mhatt = RWSA(__C, hidesize, hidesizehead, mulhead, midsize=__C.SRWSA_MIDSIZE)
        self.ffn = FFN(__C, hidesize, ffsize)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(hidesize)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(hidesize)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class SGA(nn.Module):
    def __init__(self, __C, hidesize, hidesizehead, mulhead, ffsize):
        super(SGA, self).__init__()

        self.mhatt1 = RWSA(__C, hidesize, hidesizehead, mulhead, midsize=__C.SGRWSA_SA_MIDSIZE)
        self.mhatt2 = RWSA(__C, hidesize, hidesizehead, mulhead, midsize=__C.SGRWSA_GA_MIDSIZE)
        self.ffn = FFN(__C, hidesize, ffsize)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(hidesize)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(hidesize)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(hidesize)

    def forward(self, x, y, x_mask, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt1(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.mhatt2(x, x, y, x_mask)
        ))

        y = self.norm3(y + self.dropout3(
            self.ffn(y)
        ))

        return y


class SRWSA(nn.Module):
    def __init__(self, __C, hidesize, hidesizehead, mulhead, ffsize):
        super(SRWSA, self).__init__()

        self.denceenc1 = SA(__C, hidesize, hidesizehead, mulhead, ffsize)

        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        # Get hidden vector
        x1 = self.denceenc1(x, x_mask)
        x2 = self.denceenc1(self.norm1(x1 + x), x_mask)
        x3 = self.denceenc1(self.norm2(x2 + x), x_mask)
        return x3


class SGRWSA(nn.Module):
    def __init__(self, __C, hidesize, hidesizehead, mulhead, ffsize):
        super(SGRWSA, self).__init__()

        self.dec1 = SGA(__C, hidesize, hidesizehead, mulhead, ffsize)

        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        y1 = self.dec1(x, y, x_mask, y_mask)
        y2 = self.dec1(x, self.norm1(y1 + y), x_mask, y_mask)
        y3 = self.dec1(x, y2, x_mask, y_mask)
        y3 = self.norm3(y3 + y)
        return y3


class RWSA_Layer(nn.Module):
    def __init__(self, __C):
        super(RWSA_Layer, self).__init__()

        self.srwsa = SRWSA(__C, __C.HIDDEN_SIZE, __C.HIDDEN_SIZE_HEAD, __C.MULTI_HEAD, __C.FF_SIZE)
        self.sgrwsa = SGRWSA(__C, __C.HIDDEN_SIZE, __C.HIDDEN_SIZE_HEAD, __C.MULTI_HEAD, __C.FF_SIZE)

        self.SRWSA_LAYER = __C.SRWSA_LAYER
        self.SGRWSA_LAYER = __C.SGRWSA_LAYER

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for i in range(self.SRWSA_LAYER):
            x = self.srwsa(x, x_mask)
        for i in range(self.SGRWSA_LAYER):
            y = self.sgrwsa(x, y, x_mask, y_mask)
        return x, y


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -5e4
        )
        att = F.softmax(att, dim=1)

        att_list = []
        att_list.append(
            torch.sum(att[:, :, i: i + 1] * x, dim=1)
        )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class RWSAN(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(RWSAN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(pretrained_emb)

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.RWSA_layer = RWSA_Layer(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):
        with autocast():

            # Make mask
            lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
            img_feat_mask = self.make_mask(img_feat)

            # Pre-process Language Feature
            lang_feat = self.embedding(ques_ix)
            lang_feat, _ = self.lstm(lang_feat)

            # Pre-process Image Feature
            img_feat = self.img_feat_linear(img_feat)

            # Backbone Framework
            lang_feat, img_feat = self.RWSA_layer(
                lang_feat,
                img_feat,
                lang_feat_mask,
                img_feat_mask
            )

            lang_feat = self.attflat_lang(
                lang_feat,
                lang_feat_mask
            )

            img_feat = self.attflat_img(
                img_feat,
                img_feat_mask
            )

            proj_feat = lang_feat + img_feat
            proj_feat = self.proj_norm(proj_feat)
            proj_feat = self.proj(proj_feat)

            return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)