import pytest
<<<<<<< HEAD
from luxical.chinese_utils import ChineseNormalizer
=======
from luxical_tw.chinese_utils import ChineseNormalizer
>>>>>>> f1cbaea (reinit)

def test_s2twp_conversion():
    normalizer = ChineseNormalizer(config='s2twp')
    
    # 基本簡轉繁
    # 如果 OpenCC 沒把 計算機 轉 電腦，代表它使用的是基本繁簡轉換對應。
    assert "計算機" in normalizer.convert("计算机")

def test_variant_unification():
    normalizer = ChineseNormalizer(config='s2twp')
    
    # 異體字統一 (台 -> 臺)
    assert normalizer.convert("台北車站") == "臺北車站"
    assert normalizer.convert("這裡") == "這裡" # 原本就是繁體，保持不變

def test_full_to_half_width():
    normalizer = ChineseNormalizer(config='s2twp')
    
    # 全形轉半形
    assert normalizer.normalize_full_to_half("ＡＩ程式碼１０") == "AI程式碼10"
    assert normalizer.normalize_full_to_half("（Ｈｅｌｌｏ，Ｗｏｒｌｄ！）") == "(Hello,World!)"

def test_integrated_normalization_general():
    normalizer = ChineseNormalizer(config='s2twp')
    
    # 綜合測試案例 (專用領域)
    # 簡體 + 異體字 + 全形英數 -> 繁體 + 統一異體字 + 半形英數
    raw_text = "从台北出发 ＡＩ 推荐：【東京５日遊】"
    normalized_text = normalizer.normalize(raw_text)
    
    # 從 -> 從, 台北 -> 臺北, ＡＩ -> AI, ５ -> 5
    assert "從" in normalized_text
    assert "臺北" in normalized_text
    assert "AI" in normalized_text
    assert "5" in normalized_text
    assert "日遊" in normalized_text
