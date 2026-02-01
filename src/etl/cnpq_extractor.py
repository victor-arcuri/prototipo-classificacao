import pdfplumber
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT_DIR / "data" / "pdfs" / "cnpq_taxonomy.pdf"
OUTPUT_JSON = ROOT_DIR / "data" / "taxonomies" / "cnpq_taxonomy.json"

def get_hierarchy_level(code_str):
    nums = re.sub(r'\D', '', code_str)
    if len(nums) != 8: return None
    if nums[1:7] == "000000": return 1
    elif nums[3:7] == "0000": return 2
    elif nums[5:7] == "00": return 3
    else: return 4

def clean_name(name_raw):
    name = name_raw.replace('"', '').strip()
    if name.startswith(','): name = name[1:].strip()
    return re.sub(r'\s+', ' ', name.replace('\n', ' ').replace('\r', ''))

def parse_pdf_to_json():
    if not PDF_PATH.exists():
        print(f"Arquivo não encontrado!")
        print(f"O script está procurando em: {PDF_PATH}")
        sys.exit(1)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    root = {"name": "CNPQ", "children": []}
    stack = [root]
    count_items = 0

    print(f"Lendo arquivo: {PDF_PATH.name}...")
    
    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if not text:
                    print(f"Aviso: Página {i+1} não tem texto reconhecível (pode ser imagem).")
                    continue
                    
                for line in text.split('\n'):
                    match = re.search(r'^"?(\d{8})"?\s*(.*)', line.strip())
                    
                    if match:
                        code = match.group(1)
                        raw_name = match.group(2)
                        level = get_hierarchy_level(code)
                        
                        if level is None: continue
                            
                        name = clean_name(raw_name)
                        new_node = {"name": name}
                        
                        while len(stack) > level: stack.pop()
                        
                        parent = stack[-1]
                        if "children" not in parent: parent["children"] = []
                        parent["children"].append(new_node)
                        stack.append(new_node)
                        count_items += 1

    except Exception as e:
        print(f"Erro durante o processamento do PDF: {e}")
        sys.exit(1)

    print("Gerando JSON...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(root, f, indent=2, ensure_ascii=False)
        
    print(f"Sucesso! {count_items} itens extraídos.")
    print(f"Arquivo salvo em: {OUTPUT_JSON}")

if __name__ == "__main__":
    parse_pdf_to_json()