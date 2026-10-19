# Modelos e treinamento

`training.ipynb` é o pipeline experimental de preparação, validação, treinamento e avaliação das U-Nets. `train.py`, `eval.py`, `validate_dataset.py` e `src/` permitem executar o mesmo pipeline por linha de comando. `architecture.py` é a implementação de inferência usada também pela aplicação web.

## Dados e pesos

Os dados AM4 e os checkpoints não fazem parte do repositório. Configure-os localmente e mantenha-os ignorados pelo Git:

- dados: `data/AM4/amazon/{Training,Validation,Test}`;
- pesos para a interface: `checkpoints/unet_seed_2_best.pt` e `checkpoints/attn_unet_seed_2_best.pt`.

O notebook aceita `DEFORESTATION_UNET_DIR` para localizar o diretório de trabalho. Para uma execução limpa, abra-o a partir de `models/` e ajuste as células de configuração para apontar para o diretório local do AM4.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab training.ipynb
```
