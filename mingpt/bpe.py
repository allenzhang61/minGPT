"""
bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
sequences of integers, where each integer represents small chunks of commonly
occuring characters. This implementation is based on openai's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
but was mildly modified because the original implementation is a bit confusing.
I also tried to add as many comments as possible, my own understanding of what's
going on.

bpe 是 Byte Pair Encoder（字节对编码器）的缩写。它会把任意 utf-8 字符串
转换成整数序列，每个整数代表常见字符片段中的一小块。
本实现基于 OpenAI 的 GPT-2 encoder.py，并做了少量改动，让流程更容易理解。
"""

import os
import json
import regex as re
import requests

import torch

# -----------------------------------------------------------------------------

def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually. Some bytes have their appearance preserved
    because they don't cause any trouble. These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
    bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). Instead,
    this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
    that "look nice", either in their original form, or a funny shifted character
    like 'Ā', or 'Ġ', etc.

    每个可能的字节（本质上是 0..255 的整数）都会被 OpenAI 映射到一个可视化的
    unicode 字符。某些字节原样显示没有问题，就保持原样；而像 chr(0) 这样的字节
    显示效果不好，所以会被移动到 256 之后的 unicode 区间，例如空格 32 会被映射为 'Ġ'。
    这本质上就是从字节到“好看 unicode 字符”的一一映射。
    """
    # the 188 integers that render fine in their original form and need no shifting；这 188 个整数原样显示正常，不需要平移
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] # all integers b in bs will simply map to chr(b) in the output dict；这些字节会直接映射到 chr(b)
    # now get the representations of the other 68 integers that do need shifting；接着处理另外 68 个需要平移的整数
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop；它们会映射到 chr(256+n)
    n = 0
    for b in range(2**8):
        if b not in bs:
            # if this byte is "ugly" then map it to the next available "nice" character；显示不友好的字节映射到下一个可用的“好看”字符
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d

def get_pairs(word):
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word.

    返回 word 中所有相邻元素组成的二元组集合。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges):
        # byte encoder/decoder；字节级编码器/解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        # bpe token encoder/decoder；BPE token 编码器/解码器
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        # bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab；定义 BPE 合并树的二元组合并表
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # the splitting pattern used for pre-tokenization；预分词使用的切分正则
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions <-- original openai comment
        # 原 OpenAI 注释：本应加入 re.IGNORECASE，让缩写的大写形式也能参与 BPE 合并。
        r"""
        ok so what is this regex looking for, exactly?
        python re reference: https://docs.python.org/3/library/re.html
        - the vertical bars | is OR, so re.findall will chunkate text as the pieces match, from left to right
        - '\'s' would split up things like Andrej's -> (Andrej, 's)
        - ' ?\p{L}': optional space followed by 1+ unicode code points in the category "letter"
        - ' ?\p{N}': optional space followed by 1+ unicode code points in the category "number"
        - ' ?[^\s\p{L}\p{N}]+': optional space, then 1+ things that are NOT a whitespace, letter or number
        - '\s+(?!\S)': 1+ whitespace characters (e.g. space or tab or etc) UNLESS they are followed by non-whitespace
                       so this will consume whitespace characters in a sequence but exclude the last whitespace in
                       that sequence. that last whitespace has the opportunity to then match the optional ' ?' in
                       earlier patterns.
        - '\s+': 1+ whitespace characters, intended probably to catch a full trailing sequence of whitespaces at end of string
        So TLDR:
        - we are special casing a few common apostrophe constructs ('s, 't, 're, ...) and making those into separate tokens
        - we then separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces

        这个正则到底在找什么？
        - 竖线 | 表示 OR，re.findall 会从左到右按匹配片段切分文本
        - '\'s' 会把 Andrej's 之类的词拆成 (Andrej, 's)
        - ' ?\p{L}'：可选空格 + 1 个或多个 unicode 字母
        - ' ?\p{N}'：可选空格 + 1 个或多个 unicode 数字
        - ' ?[^\s\p{L}\p{N}]+'：可选空格 + 1 个或多个非空白、非字母、非数字字符
        - '\s+(?!\S)'：匹配一段空白，但排除后面还有非空白字符的最后一个空白
        - '\s+'：兜底匹配末尾连续空白
        简而言之：先特殊处理常见撇号缩写，再把文本切成字母、数字、符号、空白几类连续片段。
        """
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        """
        this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens
        up the tree. token is a string of one individual 'word' (after regex tokenization)
        and after byte encoding, e.g. 'Ġthere'.

        这个函数会根据 self.bpe_ranks 反复合并可合并的 BPE 片段。
        token 是预分词和字节编码后的单个“词”字符串，例如 'Ġthere'。
        """
        # token is a string of one individual 'word', after byte encoding, e.g. 'Ġthere'；token 是字节编码后的单个词片段

        # memoization, for efficiency；用缓存避免重复计算
        if token in self.cache:
            return self.cache[token]

        word = tuple(token) # individual characters that make up the token, in a tuple；组成 token 的单字符元组
        pairs = get_pairs(word) # get all bigrams；获取所有相邻字符对

        if not pairs:
            return token

        while True:

            # find the next lowest rank bigram that can be merged；找到排名最高（rank 最小）的可合并二元组
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # no more bigrams are eligible to be merged；没有可继续合并的二元组
            first, second = bigram

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            # 接下来把当前 word 中所有 (first, second) 替换为一个合并后的 token：first_second。
            new_word = []
            i = 0
            while i < len(word):

                # find the next occurence of first in the sequence of current words；在当前序列中查找下一个 first
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one；如果 first 后面紧跟 second，就合并成一个 token
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second；所有匹配的二元组都已经合并完成
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # concat all words into a string, and use ' ' as the separator. Note that
        # by now all characters have been byte encoded, guaranteeing that ' ' is
        # not used in the actual data and is a 'special' delimiter character
        # 把所有片段用空格连接。由于真实数据中的字符已做字节编码，
        # 普通空格不会出现在数据片段里，因此可以安全地作为特殊分隔符。
        word = ' '.join(word)

        # cache the result and return；缓存并返回结果
        self.cache[token] = word
        return word

    def encode(self, text):
        """ string goes in, list of integers comes out；输入字符串，输出整数列表 """
        bpe_idx = []
        # pre-tokenize the input text into string tokens (words, roughly speaking)；先把文本预切分成字符串 token（大致可理解为词）
        tokens = re.findall(self.pat, text)
        # process each token into BPE integers；逐个 token 转成 BPE 整数
        for token in tokens:
            # encode the token as a bytes (b'') object；把 token 编码成 bytes 对象
            token_bytes = token.encode('utf-8')
            # translate all bytes to their unicode string representation and flatten；把每个字节翻译为 unicode 表示并拼平
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            # perform all the applicable bpe merges according to self.bpe_ranks；根据 BPE rank 执行所有可用合并
            token_merged = self.bpe(token_translated).split(' ')
            # translate all bpe tokens to integers；把 BPE token 转为整数 id
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # extend our running list of all output integers；追加到最终整数序列
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        """ debugging function, same as encode but returns all intermediate work；调试函数，和 encode 类似但会返回中间结果 """
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })
        out = {
            'bpe_idx': bpe_idx, # the actual output sequence；实际输出序列
            'tokens': tokens, # result of pre-tokenization；预分词结果
            'parts': parts, # intermediates for each token part；每个 token 片段的中间结果
        }
        return out

    def decode(self, bpe_idx):
        """ list of integers comes in, string comes out；输入整数列表，输出字符串 """
        # inverse map the integers to get the tokens；反向映射整数 id 得到 token
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes；反向字节编码，例如把 'Ġ' 还原为空格并得到 bytes
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # recover the full utf-8 string；恢复完整 utf-8 字符串
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text

def get_file(local_file, remote_file):
    """ downloads remote_file to local_file if necessary；必要时把远程文件下载到本地 """
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        with open(local_file, "wb") as f:
            f.write(response.content)

def get_encoder():
    """
    Returns an instance of the GPT BPE Encoder/Decoder
    and handles caching of "database" files.

    返回 GPT BPE 编码器/解码器实例，并处理词表等“数据库”文件的本地缓存。
    """
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    # load encoder.json that has the raw mappings from token -> bpe index；加载 token 到 BPE id 的原始映射
    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == 50257 # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token；256 个字节 token、50000 个合并 token、1 个特殊 token

    # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure；加载包含 BPE 合并规则的 vocab.bpe，也就是 BPE 树结构
    # in the form tuples (a, b), that indicate that (a, b) is to be merged to one token ab；每条规则形如 (a, b)，表示合并成 ab
    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # light postprocessing: strip the version on first line and the last line is a blank；轻量后处理：去掉首行版本信息和末尾空行
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000 # 50,000 merged tokens；50000 条合并规则

    # construct the Encoder object and return；构造并返回 Encoder
    enc = Encoder(encoder, bpe_merges)
    return enc

# -----------------------------------------------------------------------------

class BPETokenizer:
    """ PyTorch-aware class that wraps the Encoder above；面向 PyTorch 的 Encoder 包装类 """

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors='pt'):
        # PyTorch only; here because we want to match huggingface/transformers interface；这里只支持 PyTorch，以匹配 HuggingFace/transformers 接口
        assert return_tensors == 'pt'
        # single string input for now, in the future potentially a list of strings；目前只支持单个字符串，未来可扩展为字符串列表
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1；编码并创建大小为 1 的 batch 维
        idx = [self.encoder.encode(text)]
        # wrap into PyTorch tensor；包装成 PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # ensure a simple 1D tensor for now；目前要求输入是简单的一维 tensor
        assert idx.ndim == 1
        # decode indices to text；把索引解码回文本
        text = self.encoder.decode(idx.tolist())
        return text


if __name__ == '__main__':

    # here is an encoding example；下面是一个编码示例
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("Original text is:")
    print(text)
    print("First the text gets pre-tokenized, broken up into chunks, the outcome is:")
    print(r['tokens'])
    # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    print("Then we iterate over each chunk and process them in turn...")
    for part in r['parts']:
        print(part)
    # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    # (refer to the code inside Encoder.encode for what these intermediates are)；这些中间字段的含义可参考 Encoder.encode 内部代码
    print("and the final outcome is concatenating and flattening all the token_ix:")
    print(r['bpe_idx'])
    # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # this would then become the integer input sequence to the transformer；这会成为喂给 Transformer 的整数输入序列
    print("ready to feed into a Transformer!")
