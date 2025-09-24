# Manobras (NA) + Percursos NF
App Streamlit para identificar chaves NA (manobra), calcular percursos NF (montante/jusante) por alimentador e exportar XLSX.

## Rodar local
pip install -r requirements.txt
streamlit run app.py

## Uso
- Envie `Chaves.xlsx` (ou CSV) com colunas: **Name**, **Name Bloco**, **Estado Normal**, **Alimentador**.
- Filtre por Alimentador, Chave NA, ou Chave NF.
- Exporte XLSX completo ou filtrado.
