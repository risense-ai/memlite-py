"""敏感闸全规则族。"""

from memlite_py.sensitive import SensitiveError, scan


def test_clean_text_passes():
    assert scan("今天天气不错，nginx 504 超时排查完了") == []
    assert scan("密码忘了多少？我也不知道") == []
    assert scan("password 没设，先跳过") == []


def test_pem_private_key():
    t = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END"
    assert scan(t) == ["PEM私钥"]


def test_jwt():
    tok = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9P"
    assert "JWT" in scan(f"token 是 {tok}")


def test_token_prefixes():
    assert "API密钥(sk-)" in scan("密钥 sk-abcdef0123456789abcdef")
    assert scan("sk-短") == []  # 短尾巴不算
    assert "GitHub token" in scan("ghp_0123456789abcdefghijklmnopqrstuvwxyz")
    assert "GitHub PAT" in scan("github_pat_XXXX")
    assert "AWS密钥" in scan("AKIAIOSFODNN7EXAMPLE")
    assert "Slack token" in scan("xoxb-123")


def test_plaintext_password():
    assert "明文口令" in scan("password=hunter2secret")
    assert "明文口令" in scan("密码是abc123四个")
    assert "明文口令" in scan("db_pwd: x9k2m4")
    # 占位符/疑问词不误伤
    assert scan("password=<password>") == []
    assert scan("密码是什么") == []
    assert scan("密码：xxx") == []


def test_cn_id():
    # 18 位（年份段合法）
    assert "身份证号" in scan("证件号11010519491231002X")
    assert "身份证号" in scan("证件号110105195001010019")


def test_cn_phone():
    assert "手机号" in scan("联系 13812345678 找我")
    assert scan("号码 11381234567") == []  # 第二位不合法
    assert scan("长数字 138123456789012 不算") == []  # 前后带数字不中


def test_bank_card_luhn():
    # 4111111111111111 过 Luhn
    assert "银行卡号" in scan("卡号 4111111111111111")
    # 位数对但 Luhn 不过 → 不算
    assert scan("卡号 4111111111111112") == []


def test_dedup_and_order():
    t = "sk-abcdef0123456789abcdef + ghp_x + 13812345678"
    hits = scan(t)
    assert len(hits) == len(set(hits))
    assert "API密钥(sk-)" in hits and "GitHub token" in hits and "手机号" in hits


def test_sensitive_error_carries_message():
    err = SensitiveError("拒绝入库：命中敏感信息（明文口令）")
    assert "拒绝入库" in str(err)
