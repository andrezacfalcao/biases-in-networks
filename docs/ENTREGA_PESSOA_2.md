# Pessoa 2 — comparação com arquiteturas leves

**Pergunta:** qual arquitetura tem maior acurácia no pior grupo com ERM, e em qual delas o DFR mais ajuda? A referência é a **ResNet-50 da Pessoa 1**. As outras quatro redes são mais leves e mantêm o total de cinco arquiteturas.

| Modelo | Papel | Parâmetros aprox. | GFLOPs de inferência |
| --- | --- | ---: | ---: |
| ShuffleNet V2 ×0,5 | Pessoa 2; primeira a testar | 1,4 milhões | 0,04 |
| MobileNetV3-Small | Pessoa 2 | 2,5 milhões | 0,06 |
| MNASNet 0.5 | Pessoa 2 | 2,2 milhões | 0,10 |
| EfficientNet-B0 | Pessoa 2; mantida do plano anterior | 5,3 milhões | 0,39 |
| ResNet-50 | Pessoa 1; referência do artigo | 25,6 milhões | 4,09 |

Fonte: [Torchvision 0.20](https://docs.pytorch.org/vision/0.20/models.html). São estimativas dos modelos ImageNet antes da troca da última camada. GFLOPs mede operações de inferência, não o tempo real de treino no Colab. Use o primeiro treino para medir o tempo por época.

## O caminho mais simples no Colab

1. Abra o [notebook da Pessoa 2](../entrega_pessoa_2/Waterbirds_Pessoa_2_Modelos_Leves_Colab.ipynb), selecione GPU (por exemplo, T4) e rode as três células de preparação. O notebook cria um ambiente separado em Python 3.12.
2. Deixe `SEMENTES = [1]` e rode só a célula da **ShuffleNet**. Ela executa as 100 épocas de ERM, ajusta o DFR e grava as métricas. Se concluir, baixe o ZIP de progresso. Use uma semente apenas para confirmar o funcionamento.
3. Troque para `SEMENTES = [1, 2, 3, 4, 5]` e execute **uma arquitetura por vez**. A semente 1 é reaproveitada se a saída continuar disponível no mesmo ambiente. Baixe o ZIP após cada arquitetura ou bloco de sementes.
4. Se a sessão cair, guarde os ZIPs anteriores e rode apenas as sementes pendentes em uma sessão com a mesma configuração. Os ZIPs incluem os resultados brutos e CSVs por semente, suficientes para juntar a tabela mais tarde. Se a EfficientNet-B0 já estiver rodando com o protocolo anterior, deixe concluir no ambiente anterior; se tiver concluído, reaproveite seus CSVs.

Sem Colab, use o [protocolo](REPRODUCAO_WATERBIRDS.md) e rode `python -m reproduction run --architecture shufflenet_v2_x0_5 --seed 1`. Os demais nomes são `mobilenet_v3_small`, `mnasnet0_5` e `efficientnet_b0`. Omitir `--seed` executa as cinco. A Pessoa 2 pode começar antes de receber os arquivos da ResNet-50.

## Comparação e entrega

Mantenha as divisões oficiais do Waterbirds, o pré-treinamento supervisionado no ImageNet-1K, a resolução de 224 × 224, as **100 épocas de ERM**, o DFR e as mesmas sementes. Se a GPU disponível for outra, registre o hardware. A troca das arquiteturas não exige trocar a base ou diminuir a duração do protocolo.

Cada arquitetura deve ter dez linhas no `results.csv`: ERM e DFR para cinco sementes. A comparação usa **acurácia do pior grupo** como métrica principal. Para medir o ganho do DFR, calcule `pior grupo DFR − pior grupo ERM` em pontos percentuais em cada semente e reporte a média das diferenças. Mostre também a acurácia ponderada do artigo e as acurácias por grupo. A média ponderada usa as proporções do treino; a acurácia geral usa diretamente a proporção de acertos no teste.

Da Pessoa 1, receba `protocol.json`, `results.csv`, `summary.csv` e `paper_comparison.csv` de `runs/resnet50/`. Confirme as divisões e o hash dos metadados, as sementes e a configuração do treino. O hash do arquivo adaptador da ResNet-50 pode ser diferente, pois o notebook da Pessoa 1 é anterior ao cadastro dos modelos leves; o código de ERM, DFR e das métricas é o mesmo na extensão. Não combine o antigo piloto de `results/` com as novas saídas de `runs/`.

As arquiteturas leves respondem se o DFR ajuda também em modelos compactos. Esta seleção não permite concluir sobre Transformers nem sobre todas as arquiteturas. A mesma receita de ajuste faz a comparação sob condições comuns, mas pode não maximizar o desempenho de cada rede.
