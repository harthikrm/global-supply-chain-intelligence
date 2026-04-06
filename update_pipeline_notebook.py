import nbformat

notebook_paths = ["notebooks/00_data_pipeline.ipynb", "notebooks/00_data_pipeline.py"]

for path in notebook_paths:
    if not path.endswith('.ipynb'): continue
    with open(path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'display(' in cell.source:
                cell.source = cell.source.replace('display(', 'print(')
            if 'SELECT * FROM supplier_concentration_index' in cell.source and 'WHERE' not in cell.source and 'LIMIT' not in cell.source:
                cell.source = cell.source.replace('SELECT * FROM supplier_concentration_index', 'SELECT * FROM supplier_concentration_index LIMIT 10')
            if 'SELECT * FROM macro_indicators' in cell.source and 'WHERE' not in cell.source:
                cell.source = cell.source.replace('SELECT * FROM macro_indicators LIMIT 10', "SELECT * FROM macro_indicators WHERE series_id = 'BDIY' LIMIT 10")
            if 'SELECT * FROM disruption_events' in cell.source and 'WHERE' not in cell.source:
                cell.source = cell.source.replace('SELECT * FROM disruption_events', "SELECT * FROM disruption_events WHERE severity_score >= 0.7")
                
    with open(path, 'w') as f:
        nbformat.write(nb, f)

print("Updated 00_data_pipeline notebooks.")
