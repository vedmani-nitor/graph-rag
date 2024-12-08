from typing import List
from pydantic import BaseModel


class QnA(BaseModel):
    question_number: int
    question: str
    answer: str


class QnAList(BaseModel):
    qna_list: List[QnA]

