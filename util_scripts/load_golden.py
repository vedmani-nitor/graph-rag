import json
from pathlib import Path
from graphrag_poc.schema import QnA, QnAList

def load_golden_qna() -> QnAList:
    # Read the JSON file with UTF-8 encoding
    json_path = Path("artifacts/evals_data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert the data into QnA objects with enumerated question numbers
    qna_list = [
        QnA(
            question_number=i + 1,
            question=item["question"],
            answer=item["answer"]
        )
        for i, item in enumerate(data)
    ]
    
    # Create and return the QnAList object
    return QnAList(qna_list=qna_list)

if __name__ == "__main__":
    # Test the loader
    qna_list = load_golden_qna()
    print(f"Loaded {len(qna_list.qna_list)} QnA pairs")
    # Write to a file using json_dump with ensure_ascii=False
    with open("evals/input/golden_qna.json", "w", encoding="utf-8") as f:
        json.dump(
            qna_list.model_dump(),
            f,
            ensure_ascii=False,
            indent=2
        )