# Viés e Correlações Espúrias em Redes Neurais: Reprodução do *Deep Feature Reweighting* no Waterbirds e Extensão a Arquiteturas Leves

Seminário de Redes Neurais Profundas.

## O projeto

No Waterbirds, o fundo (água ou terra) está associado à classe da ave no treino. Uma rede treinada com ERM aprende esse atalho: acerta ~98% na média ponderada, mas só ~73% no pior grupo (ave aquática em terra). O projeto reproduz o artigo [Kirichenko et al., *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations* (ICLR 2023)](https://arxiv.org/abs/2204.02937), que mostra que **retreinar só a última camada** (DFR) em dados balanceados por grupo recupera a robustez.

- **Passo 1: reprodução.** ERM e DFR com ResNet-50 (linhas Base e DFR da Tabela 2, coluna Waterbirds), 100 épocas, cinco sementes, código dos autores em commit fixado.
- **Passo 2: extensão.** O mesmo protocolo em arquiteturas menores (ShuffleNetV2 ×0,5, MobileNetV3-Small, EfficientNet-B0, DenseNet-121) para ver se o DFR também ajuda em modelos compactos.
- **Piloto** (exploratório, anterior): ResNet-50 congelada, uma semente, Grad-CAM. Não faz parte dos resultados finais.

Resultados, metodologia, discussão e referências: **[RELATORIO_COMPLETO_PROJETO.md](RELATORIO_COMPLETO_PROJETO.md)**. Artigo em LaTeX: [`overleaf/`](overleaf/).

## Estrutura

```
reproduction/   runner (python -m reproduction), adaptador de arquiteturas, métricas, patch dos autores
notebooks/      00 piloto · 01 Passo 1 (Colab) · 02 Passo 2 (Colab)
scripts/pilot/  scripts do piloto (características, Grad-CAM, figuras, slides)
scripts/passo2/ agrega as saídas de runs/ e gera a figura ERM × DFR
tests/          testes unitários e de integração
results/        CSVs e figuras (results/passo2/: tabelas do Passo 2)
docs/           PDF do artigo reproduzido e slides
overleaf/       artigo em LaTeX
data/           Waterbirds v1.0 (não versionado)
```

## Como rodar

### Opção A: Google Colab (mais simples)

Abra o notebook, selecione GPU (T4) e execute as células em ordem. Ele instala o ambiente, baixa o Waterbirds e gera um ZIP com os resultados.

- Passo 1: [`notebooks/01_passo1_resnet50_colab.ipynb`](notebooks/01_passo1_resnet50_colab.ipynb)
- Passo 2: [`notebooks/02_passo2_modelos_leves_colab.ipynb`](notebooks/02_passo2_modelos_leves_colab.ipynb) (comece com `SEMENTES = [1]`, uma arquitetura por vez)

### Opção B: máquina local com GPU NVIDIA (Linux, CUDA 12.4)

1. Baixe o [Waterbirds v1.0](https://github.com/kohpangwei/group_DRO#waterbirds) e extraia em `data/waterbirds_v1.0/` (precisa de `metadata.csv` e das imagens), ou use `--data /caminho`.
2. Instale o ambiente (Python 3.11 ou 3.12):

```sh
python -m venv .venv-repro && source .venv-repro/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r reproduction/requirements.txt
git clone https://github.com/PolinaKirichenko/deep_feature_reweighting.git .upstream/deep_feature_reweighting
```

3. Verifique (sem GPU) e execute:

```sh
python -m reproduction check                                          # dados, patch e sintaxe
python -m reproduction run --architecture resnet50                    # Passo 1: 5 sementes
python -m reproduction run --architecture shufflenet_v2_x0_5 --seed 1 # Passo 2: uma semente
```

Para consolidar o Passo 2 depois das execuções: `python scripts/passo2/aggregate_results.py --output results/passo2` e `python scripts/passo2/plot_results.py --summary results/passo2/summary_all.csv --output results/passo2/comparacao.png`.

Arquiteturas: `resnet50`, `densenet121`, `efficientnet_b0`, `mobilenet_v3_small`, `shufflenet_v2_x0_5`. Sem `--seed` rodam as sementes 1 a 5; sementes concluídas são reaproveitadas. As saídas ficam em `runs/<arquitetura>/` (`protocol.json`, `results.csv`, `summary.csv`, `paper_comparison.csv`, `seed-*/`).

### Testes

```sh
python -m unittest discover -s tests -v   # unitários
python -m tests.smoke_waterbirds          # integração em CPU (96 imagens, sem valor científico)
```

## Resultados principais (ResNet-50, 5 sementes)

| Método | Pior grupo (artigo) | Pior grupo (reprodução) | Média ponderada (artigo) | Média ponderada (reprodução) |
| --- | ---: | ---: | ---: | ---: |
| ERM | 74,9 ± 2,4 | 72,9 ± 2,8 | 98,1 ± 0,1 | 98,1 ± 0,2 |
| DFR | 92,9 ± 0,2 | 91,8 ± 1,8 | 94,2 ± 0,4 | 93,8 ± 0,5 |

Detalhes, redes leves, diferenças em relação ao artigo e limitações estão no relatório.
