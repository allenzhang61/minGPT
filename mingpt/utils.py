
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping；单调但必要的日志记录准备工作 """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist；如果工作目录不存在就创建
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)；记录命令行参数（如果有）
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself；记录完整配置
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs；受 yacs 启发的轻量配置类 """
    # TODO: convert to subclass from a dict like in yacs?；是否改成类似 yacs 的 dict 子类？
    # TODO: implement freezing to prevent shooting of own foot；实现冻结，避免误改配置
    # TODO: additional existence/override checks when reading/writing params?；读写参数时增加存在性/覆盖检查？

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing；用辅助函数支持嵌套缩进的漂亮打印 """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config；返回配置的字典表示 """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32

        从命令行参数列表更新配置，通常来自 sys.argv[1:]。
        参数格式应为 `--arg=value`，并可用 . 表示嵌套属性。
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack；拆出键和值

            # first translate val into a python object；先把字符串值转换成 Python 对象
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created

                这里需要一点解释：
                - 如果 val 只是普通字符串，literal_eval 会抛出 ValueError
                - 如果 val 表示一个 Python 字面量（如 3、3.14、[1,2,3]、False、None 等），它会被创建出来
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into；找到要写入属性的目标对象
            assert key[:2] == '--'
            key = key[2:] # strip the '--'；去掉前缀 --
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists；确保该属性已经存在
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute；覆盖配置属性
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
