from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Saksham"
    age: Optional[int] = None
    email: EmailStr = None
    gpa: float = Field(ge=0, le=4, default=0.0,description="GPA of Student. It must be between 0 and 4")


new_student = {"age": 21, "email": "jrsaksham10@gmail.com", 'gpa':'3.8'}

student = Student(**new_student)

# print(type(student.gpa))

student_dict = dict(student)

print(student_dict['email'])