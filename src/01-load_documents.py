from pathlib import Path

file_path = Path("data/company_policy.txt")

with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

print(content)