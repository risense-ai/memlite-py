"""分词：中英混合、CJK 单字+bigram、词界口径。"""

from memlite_py.tokenize import is_cjk, tokenize


def test_ascii_whole_word_lowercase():
    t = tokenize("memlite 是 Python 写的")
    assert "memlite" in t
    assert "python" in t
    assert "写" in t
    assert "Python" not in t


def test_cjk_bigram_plus_unigram():
    t = tokenize("记忆系统")
    assert "记忆" in t
    assert "忆系" in t
    assert "系统" in t
    assert "记" in t  # 单字保查询可中


def test_punctuation_separates_words():
    # 全角标点是分隔符不是词（U+3000/FF00 段不进 CJK 词界）
    t = tokenize("你好，世界。ok")
    assert "你好" in t
    assert "，" not in t
    assert "世界" in t
    assert "ok" in t
    assert ",世" not in t


def test_ascii_word_not_merged_with_cjk():
    t = tokenize("nginx504超时")
    assert "nginx504" in t
    assert "超时" in t
    assert "时" in t


def test_is_cjk_range_only():
    assert is_cjk("记")
    assert not is_cjk("，")  # U+FF0C 全角逗号不在 U+4E00-U+9FFF
    assert not is_cjk("a")
    assert not is_cjk(" ")  # U+3000 全角空格不是词
