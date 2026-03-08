import pytest
import pyarrow as pa
import jieba_fast_dat
from luxical_tw.chinese_tokenization import ChineseLexicalTokenizer

def test_chinese_lexical_tokenizer_pure_python():
    # 建立一個測試用的詞庫
    vocab = {
        "從": 1,
        "臺北": 2,
        "出發": 3,
        "AI": 4,
        "推薦": 5,
        "東京": 6,
        "5": 7,
        "日遊": 8
    }
    
    tokenizer = ChineseLexicalTokenizer(vocab=vocab)
    
    # 輸入測試文本 (包含簡體、異體字、全形)
    raw_texts = pa.array(["从台北出发 ＡＩ 推薦：【東京５日遊】", None])
    
    # 執行 Tokenization
    result_chunked = tokenizer.tokenize_batch(raw_texts, batch_size=2)
    
    # 驗證
    assert len(result_chunked) == 2
    ids = result_chunked.to_pylist()[0]
    
    # 檢查轉換出的 ID 是否正確
    assert 1 in ids # 從
    assert 2 in ids # 臺北
    assert 4 in ids # AI
    assert 7 in ids # 5
    
    # 驗證 None 處理
    assert result_chunked.to_pylist()[1] is None

if __name__ == "__main__":
    # 手動驗證流程
    vocab = {"學習": 100, "計畫": 101, "旅遊": 102}
    tk = ChineseLexicalTokenizer(vocab=vocab)
    tk.add_words(["學習計畫"])
    vocab["學習計畫"] = 103
    
    res = tk.tokenize_batch(pa.array(["我想去學習計畫與執行"]), progress_bar=False)
    print(f"Result: {res.to_pylist()}")
