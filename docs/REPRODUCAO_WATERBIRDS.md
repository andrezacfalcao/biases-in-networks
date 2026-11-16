# Pessoa 1 — Reprodução de ERM e DFR no Waterbirds

**Escopo:** as linhas Base (ERM) e DFR com validação balanceada, para Waterbirds, na Tabela 2 de [Kirichenko et al., ICLR 2023](https://openreview.net/pdf/f0a7c2996eab5d56791cb5d7feebe23aa9ca27c4.pdf). O protocolo não pretende reproduzir outras linhas, bases ou figuras do artigo. Se for exigida a reprodução do artigo inteiro, será necessário ampliar este escopo.

## O que mudou em relação ao piloto

| Aspecto | Piloto | Reprodução |
| --- | --- | --- |
| Representação | ResNet-50 congelada direto do ImageNet | ResNet-50 com pesos ImageNet-1K V1, treinada de ponta a ponta no Waterbirds com ERM e congelada somente na etapa DFR |
| ERM | Regressão logística sobre características fixas | Rede completa com entropia cruzada, SGD e 100 épocas; checkpoint da última época fixado de antemão |
| DFR | Um ajuste sobre uma amostra balanceada; seleção avaliada em toda a validação, incluindo exemplos do ajuste | Regressão logística L1, escolha de C na metade reservada da validação, ajuste em dez amostras balanceadas da validação completa e média dos coeficientes |
| Média publicada | CSV do piloto usa a distribuição do teste | Acurácia dos grupos no teste ponderada pelas frequências dos grupos no treino; média geral do teste também é exportada, em outra coluna |
| Variação | Uma semente | Cinco sementes definidas de antemão (1 a 5); média, desvio padrão amostral e resultados por semente |
| Arquiteturas | Apenas ResNet-50 | ResNet-50 na reprodução e mais quatro no estudo comparativo |

A base oficial tem 4.795 imagens de treino, 1.199 de validação e 5.794 de teste. O grupo é `2 × classe + fundo` (G0 terrestre/terra, G1 terrestre/água, G2 aquática/terra, G3 aquática/água). A métrica principal é a menor acurácia entre os grupos no teste. As complementares são acurácia de cada grupo, diferença entre melhor e pior, acurácia ponderada como no artigo e acurácia geral do teste.

O treino usa imagens de 224 × 224, aumento de dados conforme o código dos autores, lotes de 32, taxa de aprendizagem e decaimento de pesos de 0,001, momento de 0,9 e taxa constante. O DFR seleciona C entre 1, 0,7, 0,3, 0,1, 0,07, 0,03 e 0,01 pela acurácia do pior grupo na metade reservada da validação. O teste não entra no ajuste do ERM, na seleção do checkpoint nem na seleção de C.

## Fonte e diferenças registradas

O runner extrai os arquivos do [commit fixado dos autores](https://github.com/PolinaKirichenko/deep_feature_reweighting/commit/9d8d5de78feafb0ba99878b8a1f91c5cd942e62a) em uma pasta temporária e aplica [o patch versionado](../reproduction/upstream.patch). O checkout original não é alterado.

- O script público usa 20 ajustes finais do DFR por padrão; o artigo descreve 10. A execução deste projeto pede explicitamente 10, mantendo a lógica dos autores para balanceamento, padronização, L1 e média dos coeficientes.
- O apêndice descreve a padronização usando os dados de reponderação. No código público, a etapa final de DFR ajusta o padronizador nas características do treino original e o aplica à validação balanceada. Este projeto conserva essa etapa do código dos autores e registra a diferença em relação ao texto.
- O treino original imprime métricas de teste por época. O patch retira essa avaliação do treino, preserva a validação e usa o checkpoint da última época. O script de avaliação roda somente depois do treino, ajustando DFR com a validação.
- O patch também explicita a semente do DFR, troca a leitura de características por um adaptador comum e grava o resultado em JSON. A média ponderada é calculada a partir das quatro acurácias, pois a média bruta exportada originalmente pelos scripts tem outra definição. A sequência de sementes 1–5 é a escolhida para esta reprodução.

## Executar

Extraia a versão 1.0 da [base pública](https://github.com/kohpangwei/group_DRO#waterbirds) em `data/waterbirds_v1.0/` ou informe `--data /caminho/waterbirds`. A pasta deve conter `metadata.csv` e as imagens que ele referencia. Use o ambiente Linux com GPU NVIDIA e driver compatível com CUDA 12.4; o primeiro treino precisa de acesso aos pesos do ImageNet.

Na raiz do projeto, em um ambiente Python 3.11 ou 3.12 novo:

```sh
python -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r reproduction/requirements.txt
git clone https://github.com/PolinaKirichenko/deep_feature_reweighting.git .upstream/deep_feature_reweighting
python -m reproduction check
python -m reproduction run --architecture resnet50
```

Se já tiver o repositório dos autores, omita o `git clone` e use `--upstream /caminho/para/o/clone` nos comandos. O perfil padrão roda as sementes 1–5 sequencialmente. Para executar por sessões, acrescente `--seed 1` (ou outro número de 1 a 5) no mesmo ambiente e preserve `runs/` em armazenamento persistente. Não execute dois comandos da mesma arquitetura na mesma pasta de saída ao mesmo tempo. O runner reaproveita sementes concluídas e gera o resumo somente quando as cinco existem.

Com o ambiente instalado, `python -m tests.smoke_waterbirds` roda um teste opcional de integração em CPU com 96 imagens reais, uma época e pesos aleatórios. Ele verifica a execução de ERM e DFR sem gerar resultados da reprodução.

## Evidência esperada

| Método | Pior grupo (artigo) | Média ponderada (artigo) |
| --- | ---: | ---: |
| ERM | 74,9 ± 2,4% | 98,1 ± 0,1% |
| DFR | 92,9 ± 0,2% | 94,2 ± 0,4% |

Em `runs/resnet50/`, preserve:

- `protocol.json`: revisão, hashes, versões, GPU, divisões e sementes.
- `results.csv`: dez linhas após as cinco sementes, com uma linha por semente e método.
- `summary.csv`: média e desvio padrão amostral das métricas para cada método.
- `paper_comparison.csv`: valor publicado, medido e diferença em pontos percentuais das métricas comparáveis.
- `seed-*/`: JSON original da avaliação, logs, comandos e checkpoint ERM.

Os números publicados acima são referência. A conclusão deve usar os valores efetivamente obtidos e explicar diferenças, sem substituir medições por valores do artigo ou do piloto.
