import re 

def extract_answer(self, text_response,  pattern: str=r"Answer: \s*(.*)"):
        match = re.search(pattern, text_response)
        if match:
                answer_text = match.group(1).strip()
                return answer_text
        else:
                return "Answer not found"