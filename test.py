import requests

pdf_url = "https://www.rd.usda.gov/sites/default/files/pdf-sample_0.pdf"
save_path = "paper.pdf"

response = requests.get(pdf_url, timeout=20)
response.raise_for_status()

with open(save_path, "wb") as f:
    f.write(response.content)

print(f"PDF saved to {save_path}")