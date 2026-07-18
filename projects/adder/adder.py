"""
Trains a GPT to add n-digit numbers.

训练一个 GPT 来做 n 位数加法。
"""

import os
import sys
import json

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
    C.system.work_dir = './out/adder'

    # data；数据配置
    C.data = AdditionDataset.get_default_config()

    # model；模型配置
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer；训练器配置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster；模型很小，可以稍微用更快的学习率

    return C

# -----------------------------------------------------------------------------

class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.

    创建 n 位数加法问题。例如 n=2 时，85 + 50 = 135 会被表示成 "8550531"：
    - 丢弃 + 和 =，只拼接两个输入数字的各位；
    - 把结果 135 反向编码成 531，这样更贴近竖式加法从低位到高位进位的顺序。

    测试时只给模型前 2n 个数字，希望它正确补全后面的 n+1 个结果数字。
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # train/test；训练集或测试集

        # split up all addition problems into either training data or test data；把所有加法问题划分到训练集或测试集
        ndigit = self.config.ndigit
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers；n 位数加法问题总数
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500；测试集最多取 20% 或 500 条
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9；词表就是数字 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        # 序列由 a、b、a+b 组成；结果位数需要 +1 以容纳进位。
        # 但最后一个 token 不会再作为输入预测下一个 token，因此总长度再 -1；这里隐含了 EOS。
        return 3*self.config.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # given a problem index idx, first recover the associated a + b；根据问题索引恢复对应的 a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b；计算加法标签 c
        c = a + b
        # encode the digits of a, b, c into strings；把 a、b、c 的数字编码成字符串
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier；反转结果，降低学习难度
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index；每个字符转成 token id
        # x will be input to GPT and y will be the associated expected outputs；x 是 GPT 输入，y 是对应的期望输出
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence；预测序列中的下一个 token
        y[:ndigit*2-1] = -1 # we will only train in the output locations. -1 will mask loss to zero；只训练输出位置，-1 会把 loss 屏蔽为 0
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any；读取默认配置，并应用命令行覆盖项
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets；构造训练集和测试集
    train_dataset = AdditionDataset(config.data, split='train')
    test_dataset  = AdditionDataset(config.data, split='test')

    # construct the model；构造模型
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object；构造训练器
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model；模型评估辅助函数
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0
        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # isolate the first two digits of the input sequence alone；只取输入序列中代表两个加数的前 2n 位
            d1d2 = x[:, :ndigit*2]
            # let the model sample the rest of the sequence；让模型补全剩余结果位
            d1d2d3 = model.generate(d1d2, ndigit+1, do_sample=False) # using greedy argmax, not sampling；使用贪心 argmax，不采样
            # isolate the last digit of the sampled sequence；取出生成序列中的结果部分
            d3 = d1d2d3[:, -(ndigit+1):]
            d3 = d3.flip(1) # reverse the digits to their "normal" order；把反向数字翻回正常顺序
            # decode the integers from individual digits；把逐位数字解码成整数
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i # manually calculate the ground truth；手工计算真实答案
            # evaluate the correctness of the results in this batch；评估当前 batch 的正确性
            correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha；传统程序和神经网络在此正面对决
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense；最多打印 5 个错误样例用于观察
                    mistakes_printed_already += 1
                    print("GPT claims that %d + %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback；每轮迭代结束时的回调
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score；同时评估训练集和测试集
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit] # if ndigit=2 we can afford the whole train set, ow no；ndigit=2 时还能完整评估训练集
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            score = train_score + test_score
            # save the model if this is the best score we've seen so far；如果分数创新高就保存模型
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode；切回训练模式
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization；开始优化训练
    trainer.run()
