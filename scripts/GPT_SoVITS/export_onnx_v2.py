import sys
sys.path.append('./')
import torch
import torchaudio
from torch import nn
from feature_extractor import cnhubert
from text import cleaned_text_to_sequence
import soundfile
import os
import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
from module.models_onnx import SynthesizerTrn, symbols_v1, symbols_v2
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
import argparse
from torch import Tensor

from AR.models.t2s_model_onnx import sample

EOS = 1024

def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class T2SEncoder(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.vits = vits
    
    def forward(self, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        x = prompt_semantic.unsqueeze(0)
        # x -> all_phoneme_ids
        # all_phoneme_ids.len
        return x

class T2SModel(nn.Module):
    def __init__(self, t2s_path, vits_model):
        super().__init__()
        dict_s1 = torch.load(t2s_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "ojbk", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        self.t2s_model.eval()
        self.vits_model = vits_model.vq_model
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        # self.t2s_model.model.top_k = torch.LongTensor([self.config["inference"]["top_k"]])
        # self.t2s_model.model.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.t2s_model = self.t2s_model.model
        self.t2s_model.init_onnx()
        self.onnx_encoder = T2SEncoder(self.t2s_model, self.vits_model)
        self.first_stage_decoder = self.t2s_model.first_stage_decoder
        self.stage_decoder = self.t2s_model.stage_decoder

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        prompts = self.onnx_encoder(ssl_content)
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        bert = bert.unsqueeze(0)
        x = torch.cat([ref_seq, text_seq], 1)
        y_len = prompts.shape[1]
        prefix_len = prompts.shape[1]
        logits, k_cache, v_cache = self.first_stage_decoder(x, prompts, bert)
        y = prompts
        samples = sample(logits, prompts,top_k=15, top_p = 1.0, temperature=1.0)[0].unsqueeze(0)

        stop = False
        for idx in range(0, 1500):
            logits, k_cache, v_cache = self.stage_decoder(y, k_cache, v_cache, y_len, idx)
            samples = sample(logits, y,top_k=15, top_p = 1.0, temperature=1.0)[0].unsqueeze(0)
            y = torch.concat([y, samples], dim=1)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if y[0, -1] == EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0
        return y[:, -idx:].unsqueeze(0)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name):
        torch.onnx.export(
            self.onnx_encoder,
            (ssl_content),
            f"onnx/{project_name}/{project_name}_t2s_encoder.onnx",
            input_names=["ssl_content"],
            output_names=["prompts"],
            dynamic_axes={
                "ssl_content": {2: "ssl_length"},
            },
            opset_version=20,
        )
        
        prompts = self.onnx_encoder(ssl_content)

        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        bert = bert.unsqueeze(0)
        x = torch.cat([ref_seq, text_seq], 1)
        y_len = prompts.shape[1]


        num_layers = self.t2s_model.num_layers
        k_cache = [torch.zeros((1, 0, 1, 512), dtype=x.dtype, device=x.device) for _ in range(num_layers)]
        v_cache = [torch.zeros((1, 0, 1, 512), dtype=x.dtype, device=x.device) for _ in range(num_layers)]

        # Export first stage decoder
        torch.onnx.export(
            self.first_stage_decoder,
            (x, prompts, bert),
            f"onnx/{project_name}/{project_name}_t2s_fs_decoder.onnx",
            input_names=["x", "prompts", "bert"],
            output_names=["logits"] + [f"k_cache_{i}" for i in range(num_layers)] + 
                         [f"v_cache_{i}" for i in range(num_layers)],
            dynamic_axes={
                "x": {1: "x_length"},
                "prompts": {1: "prompts_length"},
                "bert": {2: "bert_length"},
            },
            verbose=False,
            opset_version=20
        )
        logits, k_cache, v_cache  = self.first_stage_decoder(x, prompts, bert)
        samples = sample(logits, prompts,top_k=15, top_p = 1.0, temperature=1.0)[0].unsqueeze(0)
        y = torch.concat([prompts, samples], dim=1)
        idx = 0
        # Export stage decoder
        torch.onnx.export(
            self.stage_decoder,
            (y, k_cache, v_cache, y_len, idx),
            f"onnx/{project_name}/{project_name}_t2s_s_decoder.onnx",
            input_names=["iy"] + [f"ik_cache_{i}" for i in range(num_layers)] + 
                        [f"iv_cache_{i}" for i in range(num_layers)] + ["y_len", "idx"],
            output_names=["logits"] + [f"k_cache_{i}" for i in range(num_layers)] + 
                         [f"v_cache_{i}" for i in range(num_layers)],
            dynamic_axes={
                "iy": {1: "iy_length"},
                **{f"ik_cache_{i}": {1: "kv_length"} for i in range(num_layers)},
                **{f"iv_cache_{i}": {1: "kv_length"} for i in range(num_layers)},
            },
            verbose=False,
            opset_version=20
        )
        

class VitsModel(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        dict_s2 = torch.load(vits_path, map_location="cpu", weights_only=False)
        self.hps = dict_s2["config"]
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps["model"]["version"] = "v1"
        else:
            self.hps["model"]["version"] = "v2"
        
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        # self.vq_model.dec.remove_weight_norm()
        
    def forward(self, text_seq, pred_semantic, ref_audio):
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        return self.vq_model(pred_semantic, text_seq, refer)[0, 0]

class GptSoVits(nn.Module):
    def __init__(self, vits, t2s):
        super().__init__()
        self.vits = vits
        self.t2s = t2s
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        return self.vits(text_seq, pred_semantic, ref_audio)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content, project_name):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        torch.onnx.export(
            self.vits,
            (text_seq, pred_semantic, ref_audio),
            f"onnx/{project_name}/{project_name}_vits.onnx",
            input_names=["text_seq", "pred_semantic", "ref_audio"],
            output_names=["audio"],
            dynamic_axes={
                "text_seq": {1: "text_length"},
                "pred_semantic": {2: "pred_length"},
                "ref_audio": {1: "audio_length"},
            },
            opset_version=20,
            verbose=False
        )

class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl = cnhubert.get_model().model

    def forward(self, ref_audio_16k):
        return self.ssl(ref_audio_16k)["last_hidden_state"].transpose(1, 2)



class MyBertModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(MyBertModel, self).__init__()
        self.bert = bert_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        res = torch.cat(outputs["hidden_states"][-3:-2], -1)[0][1:-1]
        # res = torch.cat(outputs[1][-3:-2], -1)[0][1:-1]
        # return build_phone_level_feature(res, word2ph) #directly using this may cause bug and add a subgraph, use rust code
        return res


def export_bert(project_name):
    bert_path = os.environ.get(
        "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    )
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    text = "叹息声一声接着一声传出,木兰对着房门织布.听不见织布机织布的声音,只听见木兰在叹息.问木兰在想什么?问木兰在惦记什么?木兰答道,我也没有在想什么,也没有在惦记什么."
    ref_bert_inputs = tokenizer(text, return_tensors="pt")
    word2ph = []
    for c in text:
        if c in ["，", "。", "：", "？", ",", ".", "?"]:
            word2ph.append(1)
        else:
            word2ph.append(2)
    ref_bert_inputs["word2ph"] = torch.Tensor(word2ph).int()

    bert_model = AutoModelForMaskedLM.from_pretrained(
        bert_path, output_hidden_states=True,
    )
    my_bert_model = MyBertModel(bert_model)

    torch.onnx.export(
        my_bert_model,
        (
            ref_bert_inputs["input_ids"],
            ref_bert_inputs["attention_mask"],
            ref_bert_inputs["token_type_ids"],
        ),
        f"onnx/{project_name}/bert.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["bert_feature"],
        dynamic_axes={
            "input_ids": {1: "input_ids_length"},
            "attention_mask": {1: "attention_mask_len"},
            "token_type_ids": {1: "token_type_ids_len"},
        },
        opset_version=20,
        verbose=False,
    )
    print("#### exported bert ####")


def export(vits_path, gpt_path, project_name, vits_model="v2"):
    vits = VitsModel(vits_path)
    gpt = T2SModel(gpt_path, vits)
    gpt_sovits = GptSoVits(vits, gpt)
    ssl = SSLModel()
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(["n", "i2", "h", "ao3", "a1", ",", "w", "o3", "sh", "i4", "zh", "i4", "n", "eng2", "y", "u3", "y", "in1", "zh", "u4", "sh", "ou3"], version=vits_model)])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(["w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"], version=vits_model)])
    ref_bert = torch.zeros((ref_seq.shape[1], 1024)).float()
    text_bert = torch.zeros((text_seq.shape[1], 1024)).float()
    ref_audio = torch.randn((1, 48000 * 5)).float()
    ref_audio_16k = torchaudio.functional.resample(ref_audio, 48000, 16000).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio, 48000, vits.hps.data.sampling_rate).float()

    try:
        os.mkdir(f"onnx/{project_name}")
    except:
        pass

    ssl_content = ssl(ref_audio_16k).float()

    torch.onnx.export(
        ssl,
        ref_audio_16k,
        f"onnx/{project_name}/ssl.onnx",
        input_names=["ref_audio_16k"],
        output_names=["ssl_content"],
        dynamic_axes={
            "ref_audio_16k": {1: "audio_length"},
        },
        opset_version=20,
        verbose=False
    )
    export_bert(project_name)
    gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content, project_name)

    a = gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content).detach().cpu().numpy()
    soundfile.write("out.wav", a, vits.hps.data.sampling_rate)

    if vits_model == "v1":
        symbols = symbols_v1
    else:
        symbols = symbols_v2

    MoeVSConf = {
        "Folder": f"{project_name}",
        "Name": f"{project_name}",
        "Type": "GPT-SoVits",
        "Rate": vits.hps.data.sampling_rate,
        "NumLayers": gpt.t2s_model.num_layers,
        "EmbeddingDim": gpt.t2s_model.embedding_dim,
        "Dict": "BasicDict",
        "BertPath": "chinese-roberta-wwm-ext-large",
        "AddBlank": False,
    }

    with open(f"onnx/{project_name}.json", 'w') as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--export_name", type=str, required=True, help="Project Name for the exported model")
    args = parser.parse_args()

    try:
        os.mkdir("onnx")
    except:
        pass
    gpt_path = os.path.join(args.model_path, "gpt.ckpt")
    vits_path = os.path.join(args.model_path, "sovits.pth")
    with torch.no_grad():
        export(vits_path, gpt_path, args.export_name)

    # soundfile.write("out.wav", a, vits.hps.data.sampling_rate)
