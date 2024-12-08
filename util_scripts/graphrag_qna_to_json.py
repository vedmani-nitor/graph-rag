import csv
from pathlib import Path
from graphrag_poc.schema import QnA, QnAList
import json

def load_golden_qna() -> QnAList:
    # Read the CSV file with UTF-8-sig encoding to handle BOM
    csv_path = Path(r"D:\projects\graphrag-poc\evals\input\questions_and_answers_v2.csv")
    qna_list = []
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        csv_reader = csv.DictReader(f)
        for i, row in enumerate(csv_reader):
            qna = QnA(
                question_number=i + 1,
                question=row["Question"],
                answer=row["Answer"]
            )
            qna_list.append(qna)
    
    return QnAList(qna_list=qna_list)

if __name__ == "__main__":
    qna_list = load_golden_qna()
    print(f"Loaded {len(qna_list.qna_list)} QnA pairs")
    # Write to a file using json.dump with ensure_ascii=False
    with open("evals/input/graphrag_qna.json", "w", encoding="utf-8") as f:
        json.dump(
            qna_list.model_dump(),
            f,
            ensure_ascii=False,
            indent=2
        )