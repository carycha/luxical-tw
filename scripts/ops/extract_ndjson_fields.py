import json
import argparse
import logging
from pathlib import Path
from typing import Any, List, Generator
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class NDJSONProcessor:
    """
    處理 NDJSON 檔案的類別，支援欄位擷取與內容切分。
    可選擇輸出為純文字 (.txt) 或高效能訓練格式 (.parquet)。
    """

    def __init__(self, fields: List[str], max_length: int, overlap: int, batch_size: int = 100000):
        self.fields = fields
        self.max_length = max_length
        self.overlap = overlap
        self.batch_size = batch_size

    def get_nested_value(self, data: dict, path: str) -> Any:
        keys = path.split('.')
        val = data
        try:
            for key in keys:
                val = val[key]
            return val
        except (KeyError, TypeError):
            return None

    def sliding_window(self, text: str) -> Generator[str, None, None]:
        if not text:
            return
        
        text = str(text).strip()
        if len(text) <= self.max_length:
            yield text
            return

        start = 0
        step = self.max_length - self.overlap
        if step <= 0: step = self.max_length

        while start < len(text):
            chunk = text[start:start + self.max_length]
            if chunk:
                yield chunk
            start += step
            if start >= len(text):
                break

    def _save_parquet(self, data: List[str], dest_path: Path):
        """將累積的資料儲存為單一 Parquet 檔案。"""
        # 欄位名稱統一使用 'text' 以符合 Luxical 訓練資料集慣例
        table = pa.table({"text": data})
        pq.write_table(table, dest_path)

    def process(self, source_path: Path, dest_path: Path):
        if not source_path.exists():
            logger.error(f"找不到來源檔案: {source_path}")
            return

        is_parquet = dest_path.suffix.lower() == ".parquet"
        buffer = []
        count = 0
        
        logger.info(f"正在處理 {source_path.name}...")
        
        with open(source_path, 'r', encoding='utf-8') as f_in:
            # 開啟文字檔 (如果需要的話)
            f_out = None if is_parquet else open(dest_path, 'w', encoding='utf-8')
            
            try:
                for line in tqdm(f_in, desc="資料處理中"):
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        for field in self.fields:
                            value = self.get_nested_value(record, field)
                            if value is not None:
                                for chunk in self.sliding_window(str(value)):
                                    if is_parquet:
                                        buffer.append(chunk)
                                    else:
                                        f_out.write(chunk + '\n')
                        count += 1
                    except json.JSONDecodeError:
                        continue
            finally:
                if f_out:
                    f_out.close()

        # 如果是 Parquet 格式，在最後進行寫入
        if is_parquet:
            logger.info(f"正在封裝成 Parquet 檔案 (總行數: {len(buffer):,d})...")
            self._save_parquet(buffer, dest_path)
        
        logger.info(f"處理完成！共處理 {count:,d} 筆記錄，結果已存至: {dest_path}")

def main():
    parser = argparse.ArgumentParser(description="從 NDJSON 擷取指定欄位，支援產出 .txt 或 .parquet。")
    parser.add_argument("--source", type=str, required=True, help="來源 NDJSON 檔案路徑。")
    parser.add_argument("--dest", type=str, required=True, help="輸出的路徑 (.txt 或 .parquet)。")
    parser.add_argument("--fields", nargs='+', default=["title"], help="需要拉出的欄位清單 (支援點分隔)。")
    parser.add_argument("--max_length", type=int, default=512, help="每行最大長度。")
    parser.add_argument("--overlap", type=int, default=50, help="分行重疊長度。")
    parser.add_argument("--batch_size", type=int, default=100000, help="處理批次大小。")

    args = parser.parse_args()

    dest_file = Path(args.dest)
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    processor = NDJSONProcessor(
        fields=args.fields,
        max_length=args.max_length,
        overlap=args.overlap,
        batch_size=args.batch_size
    )
    
    processor.process(Path(args.source), dest_file)

if __name__ == "__main__":
    main()
