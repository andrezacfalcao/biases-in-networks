# Pessoa 1 — reprodução de ERM e DFR (ResNet-50, Waterbirds)

**Estado:** notebook pronto para execução em GPU. Ainda não há resultados de uma reprodução completa. O teste em CPU validou a integração, e a medição de tempo na GPU Apple usou apenas alguns lotes; nenhum dos dois substitui o experimento científico.

| Método | Pior grupo no artigo | Média ponderada no artigo | Nossa reprodução |
| --- | ---: | ---: | --- |
| ERM | 74,9 ± 2,4% | 98,1 ± 0,1% | Aguardando cinco sementes |
| DFR | 92,9 ± 0,2% | 94,2 ± 0,4% | Aguardando cinco sementes |

Fonte: [Kirichenko et al., ICLR 2023, Tabela 2](https://arxiv.org/html/2204.02937#S6.T2).

- [Relato da Etapa 1 e texto para o slide](ETAPA_1_WATERBIRDS.md)
- [Notebook de execução no Google Colab](Waterbirds_Pessoa_1_Colab.ipynb)
- [Protocolo e definições das métricas](../docs/REPRODUCAO_WATERBIRDS.md)
- [Guia da Pessoa 2](../docs/ENTREGA_PESSOA_2.md)

O pacote final do notebook contém `runs/resnet50/protocol.json`, `results.csv`, `summary.csv`, `paper_comparison.csv` e os registros por semente. O relatório final deve separar o valor publicado dos valores medidos e discutir a diferença. Arquivos parciais não servem como resultado final.
