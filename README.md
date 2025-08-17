# Painel EHF – Nordeste

Painel em Dash que lê dois .xlsx (previsão e atributos) e um GeoJSON municipal.
Arquivos esperados na raiz:
- previsao_ne_5dias.xlsx
- arquivo_ne_completo_preenchido.xlsx
- municipios_nordeste.geojson

## Rodar local
pip install -r requirements.txt
python app.py

## Deploy no Render
- Conecte o repositório
- Plano Free + Python
- Ele usa render.yaml

## Atualização dos dados
1) Rode sua rotina de previsão no seu PC e gere `previsao_ne_5dias.xlsx`.
2) Substitua o arquivo no repositório (commit/push).
3) O Render baixa na próxima build/redeploy automático.
