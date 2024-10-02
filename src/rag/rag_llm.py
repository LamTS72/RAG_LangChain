import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Str_OutputParser(StrOutputParser):
        def __init__(self):
                super().__init__()

        def parser(self, text):
                return self.extract_answer(text)
        
        def extract_answer(self, text_response,  pattern: str=r"Answer: \s*(.*)"):
                match = re.search(pattern, text_response, re.DOTALL)
                if match:
                        answer_text = match.group(1).strip()
                        return answer_text
                else:
                        return text_response

class RagLLM():
        def __init__(self, llm) -> None:
                self.llm = llm
                self.prompt = hub.pull("rlm/rag-prompt")
                self.str_parser = Str_OutputParser()

        def get_chain(self, retriever):
                input_data = {
                        "context": retriever | self.format_docs,
                        "question": RunnablePassthrough()
                }

                rag_chain = (
                        input_data | self.prompt | self.llm | self.str_parser
                )
                return rag_chain
        
        def format_docs(self, docs):
                return "\n\n".join(doc.page_content for doc in docs)
        