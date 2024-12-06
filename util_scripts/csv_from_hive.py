import re
import json
import pandas as pd
from typing import List, Dict
from pathlib import Path
from loguru import logger

def extract_qa_citations(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into QA pairs using a more robust pattern
    qa_pattern = r'USER:\s*(\d+)\.\s*(.*?)\s*\n\s*\n\s*HIVE:\s*(.*?)(?=\n\s*USER:|$)'
    qa_matches = re.finditer(qa_pattern, content, re.DOTALL)
    
    results = []
    
    for match in qa_matches:
        try:
            question_num = int(match.group(1))
            question_text = match.group(2).strip()
            answer_text = match.group(3).strip()
            
            # Extract citations using a more specific pattern
            citations_pattern = r'Citations:\s*\n\s*File Name:\s*(.*?)\s*\n\s*File Path:\s*(.*?)\s*\n\s*Page Number:\s*(\d+),\s*Page Number End:\s*(\d+),\s*Chunk Number:\s*(\d+)'
            citations = []
            
            for citation_match in re.finditer(citations_pattern, answer_text):
                citation = {
                    'file_name': citation_match.group(1).strip(),
                    'file_path': citation_match.group(2).strip(),
                    'page_number': int(citation_match.group(3)),
                    'page_number_end': int(citation_match.group(4)),
                    'chunk_number': int(citation_match.group(5))
                }
                citations.append(citation)
            
            # Remove citations section from answer text
            answer_text = re.sub(r'\n\s*Citations:.*$', '', answer_text, flags=re.DOTALL).strip()
            
            results.append({
                'question_number': question_num,
                'question': question_text,
                'answer': answer_text,
                'citations': citations
            })
            logger.info(f"Successfully processed question {question_num}")
            
        except Exception as e:
            logger.error(f"Error processing QA pair: {str(e)}")
            continue
    
    return results

def create_csv():
    # Read and process the file
    input_file = Path('data/hive_answers.txt')
    output_file = Path('artifacts/hive_qa.csv')
    
    logger.info(f"Processing file: {input_file}")
    qa_data = extract_qa_citations(str(input_file))
    
    if not qa_data:
        logger.error("No data was extracted from the file")
        return
    
    # Sort by question number
    qa_data.sort(key=lambda x: x['question_number'])
    
    # Create DataFrame
    df = pd.DataFrame([{
        'Question': item['question'],
        'Answer': item['answer'],
        'Citations': json.dumps(item['citations'], ensure_ascii=False)
    } for item in qa_data])
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.success(f"Created CSV file with {len(df)} questions and answers at {output_file}")

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        "artifacts/logs/csv_from_hive.log",
        rotation="500 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <white>{message}</white>"
    )
    
    create_csv()
