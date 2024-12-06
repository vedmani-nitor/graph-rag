import json
import csv

def extract_questions(json_file_path):
    try:
        # Read the JSON file with UTF-8 encoding
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions
        questions = []
        for item in data:
            questions.append(item['question'])
        
        return questions
    except UnicodeDecodeError:
        # Fallback to different encoding if UTF-8 fails
        with open(json_file_path, 'r', encoding='latin-1') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            questions.append(item['question'])
        
        return questions

def main():
    try:
        # Extract questions
        questions = extract_questions(r'D:\projects\graphrag-poc\artifacts\evals_data.json')
        
        # Create CSV file with questions and empty answers
        with open('questions_and_answers.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Question', 'Answer'])
            # Write questions with empty answers
            for question in questions:
                writer.writerow([question, ''])
                
        print("Questions and empty answer columns have been saved to 'questions_and_answers_v2.csv'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()