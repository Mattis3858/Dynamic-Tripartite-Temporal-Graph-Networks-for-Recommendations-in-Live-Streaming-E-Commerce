
ITEM_PARSING_PROMPT = """
【角色設定】
您是一位電商商品分類專家。請根據以下提供的「賣家名稱」和「商品名稱」，完成兩項任務：
1.  生成一個簡潔、清晰、標準化的商品名稱，專注於商品本身，不要數量，並去除所有無關編號、促銷詞、特殊符號。   
2.  判斷該商品最可能屬於哪個主要類別。
假如商品中包含多種產品，就回傳多個 json

【資訊輸入】
輸入文字：由賣家「{CHANNEL_NAME}」販售的商品「{item_name}」

【回傳格式】
請以 JSON 格式回傳結果，包含 'clean_description' 和 'predicted_category' 兩個鍵。

【回傳範例】
1. 範例輸入：
由賣家「宏偉時尚珠寶」販售的商品「231230LINE44-粉晶Lvv拖鞋墜0.14」
範例輸出：
{{
  "clean_description": "粉晶拖鞋造型吊墜",
  "predicted_category": "珠寶飾品"
}}

2. 範例輸入：
由賣家「大寶家具」販售的商品「2205902(精品) 大理石餐桌X3」
範例輸出：
{{
 "clean_description": "大理石餐桌",
  "predicted_category": "家具"
}}

3. 範例輸入：
由賣家「大寶家具」販售的商品「2205902(精品) 大理石餐桌X3, 2258839(絕版) 木製櫥櫃X1」
範例輸出：
{{
 "clean_description": "大理石餐桌",
  "predicted_category": "家具"
}},
{{
 "clean_description": "木製櫥櫃",
  "predicted_category": "家具"
}},
"""
