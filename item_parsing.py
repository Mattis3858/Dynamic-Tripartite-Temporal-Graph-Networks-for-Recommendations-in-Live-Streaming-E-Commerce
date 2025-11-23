from typing import List
import pandas as pd
from pydantic import ValidationError
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
from item_parsing_prompt.V1 import ITEM_PARSING_PROMPT

load_dotenv()


class ProductItem(BaseModel):
    clean_description: str
    predicted_category: str


class Products(BaseModel):
    items: List[ProductItem]


parser = JsonOutputParser(pydantic_object=Products)

LLM = ChatOllama(
    model="gemma3:4b",
    temperature=0,
)


def build_chain():
    chain = LLM
    return chain


def get_llm_answer(CHANNEL_NAME: str, item_name: str):
    chain = build_chain()
    prompt = ITEM_PARSING_PROMPT.format(CHANNEL_NAME=CHANNEL_NAME, item_name=item_name)
    response = chain.invoke(prompt)
    result = parser.parse(response.content)
    return result

def process_csv(csv_file_path: str, max_workers: int = 5) -> pd.DataFrame:

    try:
        df = pd.read_csv(csv_file_path)[0:1000]

        required_cols = ["CHANNEL_NAME", "item_name"]
        if not all(col in df.columns for col in required_cols):
            print(f"錯誤：CSV 檔案 {csv_file_path} 必須包含 'CHANNEL_NAME' 和 'item_name' 欄位。")
            return None

        print(f"成功讀取 {csv_file_path}，總共 {len(df)} 筆資料。")

        if "predicted_category" not in df.columns:
            df["predicted_category"] = ""
        if "clean_description" not in df.columns:
            df["clean_description"] = ""
        
        df["processing_error"] = "" 

        success_count = 0
        error_count = 0

        futures_to_index = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"正在提交 {len(df)} 筆任務至 {max_workers} 個 workers...")

            for index, row in df.iterrows():
                channel_name = str(row["CHANNEL_NAME"]) if pd.notna(row["CHANNEL_NAME"]) else ""
                item_name = str(row["item_name"]) if pd.notna(row["item_name"]) else ""

                if not channel_name and not item_name:
                    print(f"--- 跳過第 {index + 1} 筆 (資料為空) ---")
                    df.at[index, "processing_error"] = "Skipped (Empty Data)"
                    continue
                future = executor.submit(get_llm_answer, channel_name, item_name)
                futures_to_index[future] = index

            print("任務提交完畢，開始處理...")
            
            for future in tqdm(concurrent.futures.as_completed(futures_to_index.keys()), total=len(futures_to_index), desc="Processing rows"):
                
                index = futures_to_index[future]
                
                try:

                    result_products_obj = future.result()

                    if (
                        isinstance(result_products_obj, dict)
                        and "items" in result_products_obj
                        and result_products_obj["items"]
                    ):
                        first_item = result_products_obj["items"][0]

                        df.at[index, "predicted_category"] = first_item["predicted_category"]
                        df.at[index, "clean_description"] = first_item["clean_description"]
                        success_count += 1
                    else:
                        df.at[index, "processing_error"] = "LLM returned no items"
                        error_count += 1
                
                except ValidationError as ve:
                    print(f"處理失敗 (第 {index + 1} 筆): LLM 回傳格式錯誤。 {ve}")
                    df.at[index, "processing_error"] = f"Validation Error: {ve}"
                    error_count += 1
                except Exception as e:
                    print(f"處理失敗 (第 {index + 1} 筆): {e}")
                    df.at[index, "processing_error"] = str(e)
                    error_count += 1

        print("\n=== 所有資料處理完成 ===")
        print(f"總結：成功 {success_count} 筆，失敗 {error_count} 筆。")
        
        return df

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {csv_file_path}")
    except pd.errors.EmptyDataError:
        print(f"錯誤：CSV 檔案 {csv_file_path} 是空的。")
    except Exception as e:
        print(f"讀取 CSV 時發生未預期錯誤: {e}")
    
    return None

if __name__ == "__main__":
    INPUT_CSV = "data/cleaned_sales_data.csv"
    OUTPUT_CSV = "data/cleaned_sales_data_test.csv"

    MAX_CONCURRENT_WORKERS = 10 

    print(f"開始處理檔案: {INPUT_CSV}")
    
    processed_df = process_csv(INPUT_CSV, max_workers=MAX_CONCURRENT_WORKERS)

    if processed_df is not None:
        try:
            processed_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"\n處理完成！結果已儲存至: {OUTPUT_CSV}")
        except Exception as e:
            print(f"\n錯誤：儲存檔案 {OUTPUT_CSV} 失敗。 {e}")
    else:
        print("\n處理未完成，未產生任何輸出檔案。")

    # result = get_llm_answer("宏偉時尚珠寶", "231230LINE44-粉晶Lvv拖鞋墜0.14")
    # print(result)