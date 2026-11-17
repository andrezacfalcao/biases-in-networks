# Etapa 1 — Reprodução de ERM e DFR com ResNet-50 no Waterbirds

**Responsável:** Pessoa 1  
**Estado da etapa:** protocolo implementado e testado em escala reduzida. A execução completa começou no Colab, segundo o relato da equipe. Os resultados das cinco sementes ainda não estão disponíveis neste checkout para conferência.

## 1. Problema e objetivo

O modelo precisa distinguir **aves aquáticas de aves terrestres**. No Waterbirds, a maioria das aves aquáticas do treino aparece sobre água, e a maioria das terrestres, sobre terra. Essa associação pode levar o modelo a usar o fundo como atalho. Um modelo que dependa desse atalho pode acertar muitas imagens no geral e ainda assim errar com frequência quando vê, por exemplo, uma ave aquática sobre terra.

A pergunta desta etapa é: **retreinar apenas a última camada de uma ResNet-50 melhora a classificação do grupo em que o modelo tem mais dificuldade?** Para responder, comparamos o treinamento convencional (**ERM**) com o **DFR**, que reaproveita a mesma rede e ajusta a camada de classificação usando uma validação equilibrada por grupo.

## 2. Artigo escolhido e escopo

Usamos como referência [Kirichenko, Izmailov e Wilson, *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*, ICLR 2023](https://arxiv.org/html/2204.02937). O recorte corresponde a **duas linhas da Tabela 2**, somente na coluna do **Waterbirds**: **Base (ERM)** e **DFR com retreinamento na validação balanceada**, ambas com ResNet-50. As outras bases, os outros métodos e as análises adicionais do artigo ficaram fora desta etapa.

O Waterbirds é a base **da própria reprodução**. Na etapa seguinte, a Pessoa 2 continua com a mesma base para testar outras arquiteturas. Assim, a arquitetura muda, mas a tarefa, as divisões dos dados e o cálculo das métricas permanecem comparáveis.

## 3. Base de dados

O [Waterbirds v1.0](https://github.com/kohpangwei/group_DRO#waterbirds) combina fotos recortadas de aves do CUB com cenários do Places. Cada grupo é uma combinação da classe da ave com o tipo de fundo. Mantivemos as divisões oficiais e conferimos os metadados e a presença das imagens.

| Grupo | Classe da ave | Fundo | Treino | Validação | Teste |
| --- | --- | --- | ---: | ---: | ---: |
| G0 | Terrestre | Terra | 3.498 | 467 | 2.255 |
| G1 | Terrestre | Água | 184 | 466 | 2.255 |
| G2 | Aquática | Terra | 56 | 133 | 642 |
| G3 | Aquática | Água | 1.057 | 133 | 642 |
| **Total** | | | **4.795** | **1.199** | **5.794** |

G1 e G2 são os grupos minoritários **no treino**. O caso mais raro é o G2, com 56 exemplos, cerca de 1,2% das imagens de treino. Qual grupo fica em último no teste é uma observação a fazer por método e semente.

## 4. Como foi montado o experimento

1. **ERM — treinar a rede completa.** A ResNet-50 começa com pesos supervisionados do ImageNet-1K V1 e recebe uma camada de saída com duas classes. Todas as camadas são ajustadas no treino do Waterbirds por 100 épocas. Usamos entropia cruzada, SGD com momento 0,9, lotes de 32, taxa de aprendizado constante de 0,001 e decaimento dos pesos de 0,001. As imagens de treino passam por recortes aleatórios e inversão horizontal. O checkpoint usado é o da última época, definido de antemão.
2. **DFR — retreinar só a camada final.** Após o ERM, congelamos a rede e usamos os vetores de 2.048 características que chegam à camada final. Dividimos a validação em duas metades: em uma, balanceamos os quatro grupos e ajustamos os candidatos de regressão logística com regularização L1; na outra, escolhemos a regularização pela acurácia do pior grupo. Com a configuração escolhida, ajustamos dez classificadores em subconjuntos balanceados da validação completa e combinamos seus coeficientes e interceptos em uma única camada.
3. **Avaliar no teste.** Para cada semente, ERM e DFR são avaliados no mesmo conjunto de teste. Os dados de teste ficam fora do treino, da seleção do checkpoint e da escolha da regularização. O protocolo usa cinco sementes (1 a 5) e calcula a média e o desvio padrão amostral entre execuções.

**O que mudou em relação ao piloto:** o código inicial congelava a ResNet-50 logo após o ImageNet e treinava apenas classificadores lineares. Agora a rede inteira aprende com Waterbirds na fase ERM; só depois ela é congelada para testar o efeito do DFR. Os números do piloto ficam separados dos da reprodução.

## 5. Métricas definidas antes do treino

| Métrica | Como é calculada | O que mostra |
| --- | --- | --- |
| **Acurácia do pior grupo (WGA)** | Menor das quatro acurácias no teste | Métrica principal: desempenho no grupo em que o método mais erra |
| Acurácia por grupo | Proporção de acertos dentro de cada grupo no teste | Em quais combinações de ave e fundo o método melhora ou piora |
| Disparidade | Acurácia do melhor grupo menos a do pior, em pontos percentuais | Distância entre os extremos |
| Acurácia ponderada do artigo | Acurácias dos grupos no teste, ponderadas pelas proporções desses grupos no **treino** | Permite comparar com a coluna de média da Tabela 2 |
| Acurácia geral do teste | Número total de acertos dividido pelo total de imagens de **teste** | Desempenho no conjunto de teste como ele é; pode diferir da média publicada |

## 6. Implementação e verificação

- Fixamos uma revisão do [código público dos autores](https://github.com/PolinaKirichenko/deep_feature_reweighting/commit/9d8d5de78feafb0ba99878b8a1f91c5cd942e62a) e registramos as adaptações em um [patch versionado](../reproduction/upstream.patch).
- Registramos diferenças entre o texto e o código público: o texto descreve dez ajustes finais, enquanto a implementação usa vinte por padrão; adotamos dez. Na padronização final, mantivemos o comportamento do código público, que calcula as estatísticas no treino original, embora o apêndice descreva o uso dos dados de reponderação. Também retiramos do treino a avaliação do teste por época.
- Preparamos o [notebook para o Colab](Waterbirds_Pessoa_1_Colab.ipynb) com ambiente separado em Python 3.12, versões fixadas das bibliotecas e uma célula por semente. O notebook empacota os CSVs e registros ao final.
- Passaram seis testes automatizados das métricas e dos relatórios. Também rodamos um teste de integração ERM → DFR em CPU, com 96 imagens reais, uma época e pesos aleatórios. Esse teste confirma que as etapas se conectam e produzem saídas; suas métricas não fazem parte do resultado científico.

## 7. Referência publicada e interpretação a fazer

| Método | Pior grupo — artigo | Média ponderada — artigo | Resultado da nossa execução |
| --- | ---: | ---: | --- |
| ERM | 74,9 ± 2,4% | 98,1 ± 0,1% | Aguardando a saída das cinco sementes |
| DFR | 92,9 ± 0,2% | 94,2 ± 0,4% | Aguardando a saída das cinco sementes |

**No artigo**, o DFR ganha **18,0 pontos percentuais** no pior grupo e perde **3,9 pontos** na média ponderada em relação ao ERM. Isso ilustra por que vale olhar para as duas métricas. **Na nossa reprodução**, será preciso verificar se a mesma tendência aparece, identificar o grupo que mais mudou e relatar as diferenças em relação à referência. O DFR usa rótulos de grupo na validação; além disso, avaliar o comportamento em um único dataset não estabelece que ele se repetirá em outros problemas.

Para a entrega, usaremos `protocol.json` (dados, versões e configuração), `results.csv` (uma linha por método e semente), `summary.csv` (média e desvio padrão) e `paper_comparison.csv` (medido versus publicado). A Pessoa 2 recebe esses arquivos da ResNet-50 para unir aos resultados de ShuffleNet V2 ×0,5, MobileNetV3-Small, EfficientNet-B0 e DenseNet-121.

## Texto pronto para um slide

**Etapa 1 — Reprodução do artigo no Waterbirds**

- **Problema:** o fundo da foto pode virar um atalho para classificar a ave.
- **Método:** treinamos a ResNet-50 completa com ERM; no DFR, congelamos a rede e retreinamos só a última camada com a validação balanceada.
- **Avaliação:** cinco sementes, teste separado e acurácia do pior grupo como métrica principal.
- **Referência:** na Tabela 2 do artigo, o pior grupo passa de **74,9% (ERM)** para **92,9% (DFR)**. Substituir ou complementar essa referência com os valores da nossa execução após o Colab concluir.

**Fala sugerida:** “O modelo pode ter uma boa média mesmo errando justamente as aves que aparecem em fundos incomuns. Por isso, reproduzimos a comparação entre ERM e DFR com a mesma ResNet-50. A pergunta é se trocar só a última camada melhora o desempenho no pior grupo e qual é o custo dessa melhora na acurácia média.”

[Protocolo com comandos e versões](../docs/REPRODUCAO_WATERBIRDS.md) · [Guia da comparação para a Pessoa 2](../docs/ENTREGA_PESSOA_2.md)
