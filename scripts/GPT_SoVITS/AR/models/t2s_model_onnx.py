import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from AR.modules.transformer_onnx import LayerNorm, TransformerEncoder, TransformerEncoderLayer

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}

inf_tensor_value = torch.FloatTensor([-float("Inf")]).float()

def logits_to_probs(
    logits,
    previous_tokens=None,
    temperature: float = 1.0,
    top_k=None,
    top_p=None,
    repetition_penalty: float = 1.35,
):
    previous_tokens = previous_tokens.squeeze()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0,
            score * repetition_penalty,
            score / repetition_penalty,
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(
                sorted_logits,
                dim=-1,
            ),
            dim=-1,
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, inf_tensor_value, logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.randn_like(probs_sort)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample(logits, previous_tokens, **sampling_kwargs):
    probs = logits_to_probs(
        logits=logits,
        previous_tokens=previous_tokens,
        **sampling_kwargs,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

class OnnxEncoder(nn.Module):
    def __init__(self, ar_text_embedding, bert_proj, ar_text_position):
        super().__init__()
        self.ar_text_embedding = ar_text_embedding
        self.bert_proj = bert_proj
        self.ar_text_position = ar_text_position

    def forward(self, x, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        return self.ar_text_position(x)

class T2SFirstStageDecoder(nn.Module):
    def __init__(
        self,
        ar_audio_embedding,
        ar_audio_position,
        h,
        ar_predict_layer,
        loss_fct,
        ar_accuracy_metric,
        top_k,
        early_stop_num,
        num_layers,
    ):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
        self.h = h
        self.ar_predict_layer = ar_predict_layer
        self.loss_fct = loss_fct
        self.ar_accuracy_metric = ar_accuracy_metric
        self.top_k = top_k
        self.early_stop_num = early_stop_num
        self.num_layers = num_layers

    def forward(self, x, prompt):
        y = prompt
        x_example = x[:, :, 0] * 0.0
        y_emb = self.ar_audio_embedding(y)
        y_emb_cache = y_emb

        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)

        y_example = y_pos[:, :, 0] * 0.0
        x_attn_mask = torch.matmul(x_example.transpose(0, 1), x_example).bool()
        y_attn_mask = torch.ones_like(torch.matmul(y_example.transpose(0, 1), y_example), dtype=torch.int64)
        y_attn_mask = torch.cumsum(y_attn_mask, dim=1) - torch.cumsum(
            torch.ones_like(y_example.transpose(0, 1), dtype=torch.int64), dim=0
        )
        y_attn_mask = y_attn_mask > 0

        x_y_pad = torch.matmul(x_example.transpose(0, 1), y_example).bool()
        y_x_pad = torch.matmul(y_example.transpose(0, 1), x_example).bool()
        x_attn_mask_pad = torch.cat([x_attn_mask, torch.ones_like(x_y_pad)], dim=1)
        y_attn_mask = torch.cat([y_x_pad, y_attn_mask], dim=1)
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)

        k_cache = [torch.zeros((0, 1, 512), dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
        v_cache = [torch.zeros((0, 1, 512), dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]

        xy_dec, k_cache, v_cache = self.h(xy_pos, mask=xy_attn_mask, k_cache=k_cache, v_cache=v_cache, first_infer=True)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

        y = torch.concat([y, samples], dim=1)
        return y, k_cache, v_cache, y_emb_cache, x_example

class T2SStageDecoder(nn.Module):
    def __init__(
        self,
        ar_audio_embedding,
        ar_audio_position,
        h,
        ar_predict_layer,
        loss_fct,
        ar_accuracy_metric,
        top_k,
        early_stop_num,
        num_layers,
    ):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
        self.h = h
        self.ar_predict_layer = ar_predict_layer
        self.loss_fct = loss_fct
        self.ar_accuracy_metric = ar_accuracy_metric
        self.top_k = top_k
        self.early_stop_num = early_stop_num
        self.num_layers = num_layers

    def forward(self, y, k_cache, v_cache, y_emb, x_example):
        y_emb = torch.cat([y_emb, self.ar_audio_embedding(y[:, -1:])], 1)

        y_pos = self.ar_audio_position(y_emb)
        xy_pos = y_pos[:, -1:]
        y_example = y_pos[:, :, 0] * 0.0
        xy_attn_mask = torch.cat([x_example, y_example], dim=1)
        xy_attn_mask = torch.zeros_like(xy_attn_mask, dtype=torch.bool)

        xy_dec, k_cache, v_cache = self.h(xy_pos, mask=xy_attn_mask, k_cache=k_cache, v_cache=v_cache, first_infer=False)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

        y = torch.concat([y, samples], dim=1)
        return y, k_cache, v_cache, y_emb

class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = float(config["model"]["dropout"])
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.0, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.0, scale=False, alpha=True)
        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )
        self.top_k = torch.LongTensor([1])
        self.early_stop_num = torch.LongTensor([-1])

    def init_onnx(self):
        self.onnx_encoder = OnnxEncoder(self.ar_text_embedding, self.bert_proj, self.ar_text_position)
        self.first_stage_decoder = T2SFirstStageDecoder(
            self.ar_audio_embedding,
            self.ar_audio_position,
            self.h,
            self.ar_predict_layer,
            self.loss_fct,
            self.ar_accuracy_metric,
            self.top_k,
            self.early_stop_num,
            self.num_layers,
        )
        self.stage_decoder = T2SStageDecoder(
            self.ar_audio_embedding,
            self.ar_audio_position,
            self.h,
            self.ar_predict_layer,
            self.loss_fct,
            self.ar_accuracy_metric,
            self.top_k,
            self.early_stop_num,
            self.num_layers,
        )

    def forward(self, x, prompts, bert_feature):
        early_stop_num = self.early_stop_num
        prefix_len = prompts.shape[1]
        x = self.onnx_encoder(x, bert_feature)
        y, k_cache, v_cache, y_emb, x_example = self.first_stage_decoder(x, prompts)

        stop = False
        for idx in range(1, 1500):
            y, k_cache, v_cache, y_emb, logits, samples = self.stage_decoder(y, k_cache, v_cache, y_emb, x_example)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0
        return y, idx

    def infer(self, x, prompts, bert_feature):
        top_k = self.top_k
        early_stop_num = self.early_stop_num
        x = self.onnx_encoder(x, bert_feature)
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_example = x[:, :, 0] * 0.0
        x_attn_mask = torch.matmul(x_example.transpose(0, 1), x_example)
        x_attn_mask = torch.zeros_like(x_attn_mask, dtype=torch.bool)

        k_cache = [torch.zeros((0, 1, 512), dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
        v_cache = [torch.zeros((0, 1, 512), dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]

        stop = False
        for idx in range(1500):
            if idx == 0:
                y_emb = self.ar_audio_embedding(y)
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = torch.concat([x, y_pos], dim=1)
                y_len = y_pos.shape[1]
                x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
                y_attn_mask = F.pad(
                    torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                    (x_len, 0),
                    value=False,
                )
                xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
                xy_dec, k_cache, v_cache = self.h(xy_pos, mask=xy_attn_mask, k_cache=k_cache, v_cache=v_cache, first_infer=True)
            else:
                y_emb = torch.cat([y_emb, self.ar_audio_embedding(y[:, -1:])], 1)
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = y_pos[:, -1:]
                y_len = y_pos.shape[1]
                xy_attn_mask = torch.zeros((1, x_len + y_len), dtype=torch.bool)
                xy_dec, k_cache, v_cache = self.h(xy_pos, mask=xy_attn_mask, k_cache=k_cache, v_cache=v_cache, first_infer=False)
            
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = sample(logits[0], y, top_k=top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                break
            y = torch.concat([y, samples], dim=1)
        return y, idx