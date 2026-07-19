"""
Trains a character-level language model.

训练一个字符级语言模型。
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system；系统配置
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'
    C.system.sample_interval = 500
    C.system.sample_tokens = 500

    # data；数据配置
    C.data = CharDataset.get_default_config()

    # model；模型配置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer；训练器配置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster；模型较小，可以稍微提高学习率

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters

    产生字符级训练 batch。
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data；从文本中截取 block_size+1 个字符
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer；把每个字符编码成整数
        dix = [self.stoi[s] for s in chunk]
        # return as tensors；返回 tensor 形式的输入和目标
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any；读取默认配置，并应用命令行覆盖项
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset；构造训练数据集
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles；这里是短脚本，不必担心文件句柄泄漏
    train_dataset = CharDataset(config.data, text)

    # construct the model；构造模型
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object；构造训练器
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback；每轮迭代结束时的回调
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num > 0 and trainer.iter_num % config.system.sample_interval == 0:
            # evaluate both the train and test score；评估并生成样例文本
            model.eval()
            with torch.no_grad():
                # sample from the model...；从模型中采样生成文本
                default_context = "O God, O God!"
                # fall back to the start of the corpus if the default prompt has unseen characters；如果默认提示词含有未见字符，就退回训练文本开头
                context = default_context if all(s in train_dataset.stoi for s in default_context) else text[:min(16, len(text))]
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, config.system.sample_tokens, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model；保存最新模型
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode；切回训练模式
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization；开始优化训练
    trainer.run()
