import json

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
        questions = extract_questions('evals_data.json')
        
        # Print questions with numbers
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}\n")
        
        # Save to file with UTF-8 encoding
        with open('questions.txt', 'w', encoding='utf-8') as f:
            for i, question in enumerate(questions, 1):
                f.write(f"{i}. {question}\n\n")
                
        print("Questions have been successfully extracted and saved to 'questions.txt'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()