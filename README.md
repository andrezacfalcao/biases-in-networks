# Viés em redes neurais: correlações espúrias no Waterbirds

Em Waterbirds, o fundo de água ou terra está associado à classe da ave no treino. O projeto mede quanto essa associação prejudica os quatro grupos no teste e verifica se retreinar a última camada com **Deep Feature Reweighting (DFR)** melhora a acurácia do pior grupo.

- **Pessoa 1:** reproduzir ERM e DFR com ResNet-50 no Waterbirds, conforme a Tabela 2 de [Kirichenko et al., ICLR 2023](https://openreview.net/pdf/f0a7c2996eab5d56791cb5d7feebe23aa9ca27c4.pdf). Comece pelo [protocolo de reprodução](docs/REPRODUCAO_WATERBIRDS.md).
- **Pessoa 2:** executar o mesmo protocolo nas outras quatro arquiteturas e comparar os resultados. Comece pela [entrega para a Pessoa 2](docs/ENTREGA_PESSOA_2.md).
- **Estado:** o pipeline e o formato dos resultados estão no repositório. A confirmação numérica exige executar as cinco sementes em GPU NVIDIA/CUDA. Consulte a pasta `runs/` da execução para obter os resultados. As referências publicadas e os números do piloto não são resultados da reprodução.

## Verificar a preparação

Na raiz do repositório, com Python 3.11 ou 3.12, Git e o Waterbirds v1.0 extraído:

```sh
# Só se ainda não houver o checkout dos autores:
git clone https://github.com/PolinaKirichenko/deep_feature_reweighting.git .upstream/deep_feature_reweighting
python -m reproduction check
python -m unittest discover -s tests -v
```

O comando `check` não exige GPU ou PyTorch. Verifica as divisões, a presença das imagens e se a adaptação se aplica à revisão fixada do código dos autores. Para instalar o ambiente e executar `python -m reproduction run`, siga o [protocolo](docs/REPRODUCAO_WATERBIRDS.md).

## Material do piloto

O [notebook](waterbirds_bias_reduction.ipynb), os [slides](seminario_vies_redes_neurais.pptx), o [relatório anterior](RELATORIO_COMPLETO_PROJETO.md) e os [resultados em CSV](results/waterbirds_model_comparison.csv) documentam um estudo exploratório com uma ResNet-50 congelada logo após o pré-treinamento no ImageNet. O piloto compara classificadores lineares com ERM, reponderação e adaptações de DFR e JTT; pode apoiar a apresentação do problema e dos mapas Grad-CAM. Os resultados do novo protocolo são gravados em outro diretório.
