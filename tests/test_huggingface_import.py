"""
Ensure that we can load huggingface/transformer GPTs into minGPT

确保可以把 huggingface/transformers 的 GPT 权重加载到 minGPT 中。
"""

import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
# -----------------------------------------------------------------------------

class TestHuggingFaceImport(unittest.TestCase):

    def test_gpt2(self):
        model_type = 'gpt2'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prompt = "Hello!!!!!!!!!? 🤗, my dog is a little"

        # create a minGPT and a huggingface/transformers model；分别创建 minGPT 和 HuggingFace 模型
        model = GPT.from_pretrained(model_type)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # init a HF model too；也初始化一个 HF 模型

        # ship both to device；把两个模型移动到同一设备
        model.to(device)
        model_hf.to(device)

        # set both to eval mode；都切换到评估模式
        model.eval()
        model_hf.eval()

        # tokenize input prompt；对输入提示词做分词
        # ... with mingpt；使用 minGPT 分词器
        tokenizer = BPETokenizer()
        x1 = tokenizer(prompt).to(device)
        # ... with huggingface/transformers；使用 HuggingFace 分词器
        tokenizer_hf = GPT2Tokenizer.from_pretrained(model_type)
        model_hf.config.pad_token_id = model_hf.config.eos_token_id # suppress a warning；设置 pad_token_id 以消除告警
        encoded_input = tokenizer_hf(prompt, return_tensors='pt').to(device)
        x2 = encoded_input['input_ids']

        # ensure the logits match exactly；确保 logits 精确匹配
        logits1, loss = model(x1)
        logits2 = model_hf(x2).logits
        self.assertTrue(torch.allclose(logits1, logits2))

        # now draw the argmax samples from each；分别用 argmax 方式生成样本
        y1 = model.generate(x1, max_new_tokens=20, do_sample=False)[0]
        y2 = model_hf.generate(x2, max_new_tokens=20, do_sample=False)[0]
        self.assertTrue(torch.equal(y1, y2)) # compare the raw sampled indices；比较原始生成索引

        # convert indices to strings；把索引转回字符串
        out1 = tokenizer.decode(y1.cpu().squeeze())
        out2 = tokenizer_hf.decode(y2.cpu().squeeze())
        self.assertTrue(out1 == out2) # compare the exact output strings too；同时比较最终输出字符串

if __name__ == '__main__':
    unittest.main()
