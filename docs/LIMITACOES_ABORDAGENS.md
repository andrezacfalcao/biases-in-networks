# Limitações das abordagens implementadas

Este documento audita o que o repositório realmente faz em ERM e DFR, e não
apenas o que o artigo descreve. As limitações estão separadas por origem:
desenho experimental, ERM, DFR, implementação, métricas e comparação com o
artigo. O objetivo é evitar que uma diferença de receita ou de código seja
interpretada como diferença entre arquiteturas.

## 1. Desenho experimental

1. **Uma única receita para cinco arquiteturas.** Todas usam 224 × 224, lote 32,
   SGD com momento 0,9, `lr = 0,001` constante, `weight_decay = 0,001`, 100
   épocas e a mesma augmentação. A receita foi escolhida para a ResNet-50 no
   artigo. Redes com 0,34 M (ShuffleNet V2 ×0,5) ou 1,52 M (MobileNetV3-Small)
   parâmetros podem ficar subtreinadas nesse orçamento, enquanto redes maiores
   podem sobreajustar. O experimento mede “mesma receita”, não o melhor
   desempenho possível de cada arquitetura.
2. **Checkpoint fixo na última época, sem seleção.** O artigo fixa a última
   época para evitar escolher o checkpoint com o teste. Isso é correto, mas
   torna o resultado sensível a quando o pior grupo atinge o pico. Não há early
   stopping nem seleção por validação; `--eval_freq 1` apenas registra a
   validação.
3. **Cinco sementes e uma única divisão oficial.** As sementes 1–5 variam
   inicialização e ordem dos lotes, mas não a divisão treino/validação/teste.
   A incerteza de amostragem do conjunto de teste não é estimada.
4. **Escolha das arquiteturas a posteriori.** A replicação original é a
   ResNet-50. As quatro redes adicionais foram escolhidas depois; a comparação
   entre elas é uma análise secundária e exploratória, não uma hipótese
   pré-registrada.
5. **Uma única base.** Waterbirds é um benchmark sintético construído colando
   aves do CUB em fundos do Places. A correlação espúria é induzida e controlada;
   os resultados não garantem o mesmo comportamento em dados reais, outras
   correlações ou outros grupos.
6. **Sem teste estatístico pareado.** O resumo traz média e desvio padrão
   amostral de cinco sementes. Não há intervalo de confiança, teste pareado por
   semente ou correção para múltiplas comparações.

## 2. ERM

1. **O ERM otimiza a média, não o pior grupo.** A classe 0 domina o treino
   (3.682 de 4.795 imagens; 76,8%). O gradiente médio favorece o grupo
   majoritário e permite que os grupos minoritários sejam sacrificados. Isso é
   o fenômeno em estudo, não um defeito acidental.
2. **Sem rebalanceamento de classes, grupos ou fundos.** Não há pesos de perda
   nem amostragem balanceada na etapa ERM. É a linha de base correta do artigo,
   mas significa que a rede pode aprender quase só o fundo.
3. **Acurácia média alta pode esconder o colapso.** Um modelo que prevê sempre
   “ave aquática” obtém cerca de 77,8% de acurácia no teste e 0% em dois grupos.
   Por isso a métrica principal é o pior grupo, e a média ponderada não deve ser
   lida isoladamente.
4. **Ajuste fino completo com a mesma taxa de aprendizagem.** A rede inteira,
   inclusive o backbone pré-treinado, é ajustada com `lr = 0,001`. Em redes
   pequenas isso pode destruir características úteis do ImageNet antes de a
   cabeça aprender; em redes grandes pode ser insuficiente. Não há
   layer-wise learning rate decay.
5. **Lote 32 e BatchNorm.** Redes com muitas camadas BatchNorm
   (MobileNetV3-Small, DenseNet-121) estimam estatísticas com 32 imagens por
   lote. Isso adiciona ruído e pode afetar a comparação com a ResNet-50.
6. **Augmentação única.** `RandomResizedCrop(0.7–1.0)` e flip horizontal são
   aplicados a todas as redes. Não há AutoAugment, RandAugment, Mixup ou corte
   específico por arquitetura; o pré-treino de cada família usou receitas
   diferentes.
7. **Sem calibração nem análise de confiança.** A avaliação mede acurácia por
   grupo, não calibração, margem ou taxa de erro por classe. Um modelo pode
   acertar o pior grupo por motivos diferentes dos que o DFR pretende corrigir.

## 3. DFR

1. **Exige rótulos de grupo na validação.** O balanceamento usa `y` e `place`
   para formar os quatro grupos. Em uma aplicação real, o fundo ou o atributo
   sensível pode não estar anotado; o método não se aplica sem essa informação.
2. **Treina com 532 exemplos, não com o treino completo.** Com
   `--balance_dfr_val` e `--notrain_dfr_val` nos valores padrão, o DFR usa
   apenas a validação balanceada: 4 grupos × 133 = 532 exemplos. O conjunto de
   treino não entra. Para 1.024–1.280 características, isso é um problema de
   regressão com mais variáveis do que amostras; a regularização L1 e a escolha
   de `C` fazem todo o trabalho.
3. **Descarta mais da metade da validação.** A validação tem 467, 466, 133 e 133
   exemplos por grupo. O balanceamento para o mínimo usa 133 de cada grupo e
   descarta 667 exemplos. Uma alternativa seria ponderar todos os exemplos; não
   é o que o código faz.
4. **A validação não é um holdout do DFR.** Metade da validação treina e a outra
   metade escolhe `C` e os pesos de classe; depois o DFR é retreinado na
   validação completa balanceada. O teste é separado, mas não existe um segundo
   conjunto de validação para estimar a generalização do DFR. O número final é
   otimista em relação a um holdout limpo, embora não haja vazamento do teste.
5. **Seleção de hiperparâmetros com uma única divisão.** `dfr_on_validation_tune`
   usa `num_retrains=1`: uma única metade aleatória da validação para treinar e
   a outra para escolher entre 7 valores de `C` × 14 opções de peso de classe
   (98 combinações). Os dez retreinos finais reduzem a variância dos
   coeficientes, mas não a variância da escolha de hiperparâmetros.
6. **Padronização divergente do texto e entre as duas etapas.** Na avaliação
   final, o `StandardScaler` é ajustado nas características do treino original e
   aplicado à validação balanceada. Na sintonia, ele é ajustado na metade da
   validação usada para treinar. O apêndice descreve o ajuste nos dados de
   reponderação. O repositório mantém o código dos autores e registra a
   diferença; ainda assim, as médias e os desvios vistos pelo classificador não
   são os da validação balanceada em nenhuma das duas etapas.
7. **Só a última camada.** Se a representação congelada não contiver
   características centrais da ave, o DFR não as cria. Em redes pequenas
   subtreinadas, o teto do DFR pode ser baixo; em redes que aprenderam o fundo
   de forma extrema, a camada linear pode apenas reorganizar o atalho.
8. **Trade-off com a média.** O DFR pode ganhar no pior grupo e perder na média
   ponderada, como no artigo (74,9 → 92,9 no pior grupo; 98,1 → 94,2 na média).
   Reportar só o ganho no pior grupo esconde o custo.
9. **Dez retreinos, não vinte.** O script público usa 20 por padrão; o artigo
   descreve 10 e o projeto força 10. A média dos coeficientes reduz ruído, mas
   o número é uma escolha de implementação.
10. **Grade de `C` e L1 fixos.** A grade `[1, 0,7, 0,3, 0,1, 0,07, 0,03, 0,01]`
    e a penalidade L1 vêm do código dos autores. Não há busca em L2, elastic net
    ou validação cruzada interna.
11. **Peculiaridade do código dos autores.** Em `dfr_on_validation_tune`, quando
    `add_train=True`, os primeiros `n_train` exemplos do treino são usados sem
    embaralhar (`all_embeddings["train"][:n_train]`), o que depende da ordem do
    CSV. Na configuração padrão do projeto `add_train=False`, então esse caminho
    não é exercitado; ainda assim, é uma armadilha para quem alterar as opções.
12. **O DFR não corrige o viés de anotação.** Se o rótulo de fundo ou de espécie
    estiver errado, o balanceamento e a métrica de pior grupo herdam o erro.

## 4. Implementação e engenharia

1. **O log chama a validação de “Test results”.** O `train_classifier.py`
   original avaliava teste a cada época; o patch remove o teste, mas mantém o
   rótulo impresso. Quem lê `seed-*/erm/log.txt` pode confundir a validação com
   o teste. O teste real só é avaliado no `dfr_evaluate_spurious.py`.
2. **`tmp_checkpoint.pt` não é reaproveitado.** O treino salva um checkpoint a
   cada época, mas o runner só reutiliza `final_checkpoint.pt`. Uma interrupção
   no meio do ERM recomeça as 100 épocas; só a interrupção durante o DFR
   aproveita o ERM concluído.
3. **Manifesto incompleto.** `protocol.json` registra commit, patch, adaptador,
   versões, GPU, divisões e sementes, mas não a grade de `C`, os pesos de classe,
   `num_retrains`, `eval_freq` ou o número de workers. Esses valores estão em
   `commands.txt` e no código, mas não em um único lugar auditável.
4. **Qualquer mudança de ambiente invalida a pasta.** O runner exige igualdade
   exata do manifesto para retomar. É bom para integridade, mas impede continuar
   uma execução após atualizar PyTorch, CUDA ou o adaptador; é preciso começar
   outro `--output`.
5. **Dependência de GPU e de versões específicas.** O treino exige CUDA. Nesta
   máquina, a RTX 5070 (Blackwell, compute capability 12.0) não roda a build
   `cu124` do PyTorch 2.5.1; foi necessário usar `torch 2.11.0+cu128` e
   `torchvision 0.26.0`. A diferença de versões e hardware deve ser registrada
   ao comparar com a ResNet-50 da Pessoa 1.
6. **Determinismo parcial.** `set_seed` fixa `torch`, CUDA, NumPy e
   `cudnn.deterministic=True`, mas `DataLoader` com quatro workers, operações
   atômicas na GPU e algumas camadas não têm garantia bit a bit. As sementes
   reduzem, mas não eliminam, a variação entre execuções.
7. **Execução sequencial e sem paralelismo entre arquiteturas.** O runner roda
   uma arquitetura por vez e as sementes em sequência; não há fila, retomada
   automática de `tmp_checkpoint` ou paralelismo. Em uma GPU pequena, o gargalo
   é o carregamento de imagens, não a GPU.
8. **Resultados brutos fora do Git.** `/runs/` está no `.gitignore`; os CSVs
   resumidos precisam ser exportados para a entrega. Isso evita versionar
   checkpoints, mas separa o resultado do código que o gerou.
9. **Adaptador cobre só cabeças lineares únicas.** `MODEL_SPECS` substitui um
   `Linear` no caminho indicado. Modelos com cabeças auxiliares, *pooling*
   diferente ou saída multicabeça exigiriam mudança no adaptador e no gancho de
   características.
10. **Sem *mixed precision* nem *gradient accumulation*.** O treino usa FP32 e
    lote 32 fixo. Não há problema de memória para as cinco redes nesta GPU, mas
    o custo é maior do que o necessário.

## 5. Métricas e inferência

1. **O pior grupo é um mínimo de quatro estimativas ruidosas.** O grupo
   minoritário do teste tem 642 exemplos. O erro-padrão binomial de uma
   acurácia de 50% é cerca de 2,0 pontos percentuais; para 90%, cerca de 1,2.
   O mínimo entre quatro grupos é enviesado para baixo e muda de grupo entre
   sementes.
2. **A média ponderada é dominada pelo grupo majoritário.** Os pesos do treino
   são 3.498, 184, 56 e 1.057; o grupo 0_0 responde por 73% da média. Uma
   melhora de 10 pontos no pior grupo pode quase não aparecer na média.
3. **`group_gap_pp` é apenas máximo menos mínimo.** Não diz qual grupo piorou,
   nem se a piora foi no grupo minoritário ou no majoritário. As acurácias por
   grupo precisam ser lidas junto.
4. **Sem intervalo de confiança.** Com cinco sementes, o intervalo de 95% para
   a média usa aproximadamente `média ± 1,24 × desvio-padrão` (t de Student com
   4 graus de liberdade). Um desvio de 2 pontos vira uma incerteza de ±2,5
   pontos; diferenças menores entre arquiteturas não são distinguíveis.
5. **A incerteza do conjunto de teste não entra no desvio entre sementes.** As
   cinco sementes usam o mesmo teste; o desvio mede variação de treino, não
   erro de amostragem do teste. Para comparar arquiteturas, seria necessário um
   bootstrap pareado nas imagens de teste ou vários testes independentes.
6. **Múltiplas comparações.** Quatro arquiteturas × dois métodos × quatro
   grupos × várias métricas, sem correção. Os resultados são exploratórios;
   não se deve escolher o “melhor” número e tratá-lo como confirmatório.
7. **Acurácia não mede calibração nem justiça além dos quatro grupos.** Um
   modelo pode ter o mesmo pior grupo e distribuir erros de forma diferente
   entre espécies, fundos ou tipos de imagem.
8. **Ruído de rotulagem.** Os grupos vêm do atributo `place`, construído a
   partir da colagem de aves em fundos do Places. Segmentação, oclusão e
   artefatos de composição podem fazer o rótulo de fundo não corresponder ao que
   a rede vê. Parte do erro do pior grupo pode ser ruído, não viés aprendido.

## 6. Comparação entre arquiteturas e com o artigo

1. **Só a ResNet-50 tem referência publicada nesta tabela.** O artigo reporta
   ERM 74,9 ± 2,4% e DFR 92,9 ± 0,2% no pior grupo, e 98,1 ± 0,1% / 94,2 ± 0,4%
   na média ponderada. Para ShuffleNet, MobileNetV3-Small, EfficientNet-B0 e
   DenseNet-121 não há número publicado com este protocolo; a comparação é
   interna.
2. **Pesos pré-treinados não são equivalentes.** Todos usam ImageNet-1K V1 do
   torchvision, mas as receitas de pré-treino diferem por família (resolução,
   augmentação, otimizador, número de épocas). A comparação confunde capacidade
   arquitetural com qualidade e viés do pré-treino.
3. **Resolução e augmentação iguais não são neutras.** 224 × 224 e o mesmo
   `RandomResizedCrop` podem favorecer redes com certo campo receptivo e
   *inductive bias*. O “melhor” setup de cada arquitetura não foi buscado.
4. **A réplica é parcial.** O artigo tem outras linhas (Group DRO, JTT, DFR
   com outras seleções) e outras bases. O repositório reproduz apenas as linhas
   Base (ERM) e DFR com validação balanceada no Waterbirds; a Tabela 2 completa
   não é reproduzida.
5. **O DFR do artigo usa a representação da ResNet-50.** Transferir a conclusão
   “retreinar a última camada basta” para redes pequenas exige verificar se a
   representação delas contém as características centrais. É exatamente o que
   os resultados por arquitetura devem mostrar, não presumir.
6. **Sem controle de hardware.** A Pessoa 1 pode rodar com outra GPU, driver,
   versão de PyTorch e número de workers. O `protocol.json` registra parte
   disso; a comparação final precisa conferir os manifestos antes de juntar os
   números.

### 6.1 Evidência observada nas quatro arquiteturas (cinco sementes)

A comparação completa ilustra quase todas as limitações acima. Em todas as
redes, o pior grupo no **treino** é muito maior que no **teste**: ~100% na
MobileNet, EfficientNet e DenseNet (99,5–100%) e 89,3–94,6% na ShuffleNet,
contra 35–79% no teste. O ERM ajusta os 56 exemplos minoritários do treino e
não generaliza para os 642 do teste. O problema é o atalho, não a otimização.

Na ShuffleNet V2 ×0,5:

- ERM: pior grupo no teste **36,9 ± 1,2%**, média ponderada **95,3 ± 0,1%**.
  No treino, o pior grupo chega a **89,3–94,6%**. A rede praticamente decora os
  56 exemplos minoritários do treino e não generaliza para o teste.
- DFR: pior grupo **76,7 ± 5,6%**, média ponderada **82,3 ± 1,2%**. O ganho no
  pior grupo é **+39,8 ± 4,5 pontos**, mas o custo na média é **−13,0 ± 1,3
  pontos**. É um trade-off muito maior que o da ResNet-50 no artigo (−3,9).
- O pior grupo do ERM é sempre o 1_0 (ave aquática em terra). Depois do DFR, o
  pior grupo passa a ser 0_1 ou 0_0, ambos majoritários. O DFR não elimina o
  problema: ele desloca o gargalo para outro grupo.
- O `C` escolhido varia de 0,03 a 1,0 entre sementes. Com 532 exemplos e 1.024
  características, a seleção de hiperparâmetros é instável; o desvio de 5,6
  pontos no pior grupo do DFR reflete isso.
- Na semente 1, a melhor validação do ERM foi 42,9% no pior grupo na época 96,
  mas o checkpoint fixo da época 99 tem 36,1%. Na semente 2, a época 99 foi a
  melhor. A regra da última época adiciona variância, para melhor ou para pior.

Na MobileNetV3-Small, o mesmo padrão aparece com menos custo: ERM **41,8 ±
2,1%** no pior grupo e **95,7 ± 0,1%** na média ponderada; DFR **82,7 ± 1,5%**
no pior grupo e **85,0 ± 0,9%** na média. O ganho é **+40,9 ± 1,6 pontos** e o
custo é **−10,7 ± 0,9 pontos**. O pior grupo do ERM também é sempre o 1_0; no
DFR, passa a ser 0_0 ou 0_1. O `C` escolhido varia de 0,1 a 1,0.

Na EfficientNet-B0, intermediária, o ERM já chega a **62,0 ± 1,5%** no pior
grupo e **96,7 ± 0,1%** na média ponderada. O DFR sobe para **85,2 ± 0,3%** e
**90,3 ± 0,5%**: ganho de **+23,1 ± 1,3 pontos** no pior grupo com custo de
apenas **−6,4 ± 0,5 pontos** na média. O `C` varia de 0,3 a 1,0 e o pior grupo
do DFR é sempre 0_1. A rede maior precisa de menos “conserto” e paga menos por
ele; a receita comum, portanto, não é neutra em relação à capacidade.

Na DenseNet-121, a maior das quatro, o ERM chega a **73,8 ± 2,9%** no pior grupo
e **97,7 ± 0,1%** na média; o DFR vai a **92,2 ± 0,5%** e **93,8 ± 0,3%**. O
ganho é **+18,4 ± 3,0 pontos** e o custo é **−3,9 ± 0,3 pontos** — praticamente
o mesmo trade-off da ResNet-50 no artigo (−3,9). O `C` varia de 0,3 a 1,0 e o
pior grupo do DFR é 0_1 em quatro das cinco sementes. Quanto maior a rede, menor
o ganho relativo do DFR e menor o custo; o teto do ganho é a representação que o
ERM já aprendeu.

A comparação completa, com as tabelas e a figura, está em
[`entrega_pessoa_2/LEIA-ME.md`](../entrega_pessoa_2/LEIA-ME.md).

## 7. Limitações específicas do piloto em `results/`

O piloto não é a reprodução do artigo e não deve ser somado aos números de
`runs/`. As principais diferenças são:

1. **Representação congelada.** O piloto extrai características de uma ResNet-50
   pré-treinada no ImageNet e treina apenas classificadores lineares. O ERM do
   artigo ajusta a rede inteira no Waterbirds; um probe linear não mede o mesmo
   fenômeno.
2. **Pesos diferentes.** O piloto usa `ResNet50_Weights.DEFAULT` (ImageNet-1K
   V2); a reprodução usa `IMAGENET1K_V1`. A representação e o pré-treino não são
   os mesmos.
3. **Uma semente.** O piloto usa a semente 42 e não reporta média nem desvio. A
   reprodução usa as sementes 1–5 definidas de antemão.
4. **Sem augmentação no treino.** As características são extraídas com
   `Resize(256) + CenterCrop(224)`; não há `RandomResizedCrop` nem flip. O
   classificador linear vê uma representação fixa e “limpa”.
5. **Seleção de `C` com vazamento.** O DFR do piloto ajusta cada candidato na
   validação balanceada e avalia o pior grupo na validação inteira, que inclui
   os 532 exemplos usados no ajuste. O `C` escolhido é otimista. A reprodução
   reserva metade da validação para escolher e a outra metade para treinar.
6. **DFR com L2 e uma única amostra.** O piloto usa a regressão logística padrão
   (L2), um único subconjunto balanceado e nenhuma média de retreinos. O artigo
   e a reprodução usam L1, seleção de `C` e média de 10 retreinos.
7. **JTT e reweighting lineares.** As duas alternativas do piloto também são
   cabeças lineares sobre características congeladas; o conjunto de erros do JTT
   vem do probe linear, não de uma rede treinada de ponta a ponta.
8. **Média com pesos do teste.** O CSV do piloto usa a distribuição do teste
   para a média; a reprodução usa as frequências dos grupos no treino, como no
   artigo. As duas colunas não são intercambiáveis.
9. **Grad-CAM ilustrativo.** Os mapas do piloto ajudam a visualizar o atalho,
   mas não são métrica de comparação nem prova de que o modelo usa o fundo.

## 8. Onde verificar no código

| Tema | Referência |
| --- | --- |
| Hiperparâmetros do ERM e do DFR | `reproduction/__main__.py`, função `commands` |
| Remoção da avaliação de teste no treino e `num_retrains=10` | `reproduction/upstream.patch` |
| Cabeças substituídas e arquiteturas disponíveis | `reproduction/architectures.py` |
| Hash e contagens do `metadata.csv` | `reproduction/metrics.py`, `read_metadata` |
| Métricas e resumo por semente | `reproduction/metrics.py`, `result_rows` e `summarize` |
| Protocolo narrativo e referência publicada | `docs/REPRODUCAO_WATERBIRDS.md` |
| Agregação e intervalos de confiança | `scripts/aggregate_results.py` |
| Código dos autores (commit fixado) | `.upstream/deep_feature_reweighting`, commit `9d8d5de` |

## 9. Como reportar sem esconder essas limitações

- Apresentar, para cada arquitetura e método, média ± desvio do pior grupo e a
  média ponderada, com o grupo que foi o pior em cada semente.
- Calcular o efeito do DFR por semente (`DFR − ERM`), reportar a média e o
  desvio dessas diferenças; não comparar médias independentes.
- Dizer explicitamente que a comparação entre arquiteturas usa a mesma receita,
  não o melhor resultado de cada uma.
- Tratar a ResNet-50 como a única comparação direta com o artigo e as demais
  como análise exploratória.
- Registar versões, GPU, commit, hash do `metadata.csv` e eventuais desvios.
- Não misturar os números do piloto em `results/` com os novos de `runs/`.
- Se o DFR ganhar no pior grupo e perder na média, mostrar as duas colunas.
- Se uma rede pequena colapsar no ERM, discutir se o DFR teve representação
  suficiente; não atribuir o resultado apenas ao método.
