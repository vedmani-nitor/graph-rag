from pathlib import Path
from graphrag_poc.schema import QnA, QnAList
import json

def load_json_qna(file_path: Path) -> QnAList:
    """Load QnA data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return QnAList.model_validate(data)

def load_all_data() -> dict[str, QnAList]:
    """Load all QnA datasets."""
    base_path = Path("evals/input")
    
    datasets = {
        "golden": base_path / "golden_qna.json",
        "hive": base_path / "hive_qna.json",
        "graphrag": base_path / "graphrag_qna.json"
    }
    
    return {
        name: load_json_qna(path)
        for name, path in datasets.items()
    }

# Load all datasets
all_datasets = load_all_data()

# You can access individual datasets like this:
golden_data = all_datasets["golden"]
hive_data = all_datasets["hive"]
graphrag_data = all_datasets["graphrag"]

# Print some basic statistics
for name, dataset in all_datasets.items():
    print(f"{name} dataset contains {len(dataset.qna_list)} QnA pairs")

# print(golden_data.qna_list[0].question)
# print(graphrag_data.qna_list[0].question)
# print(hive_data.qna_list[0].question)

