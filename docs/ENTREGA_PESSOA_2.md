# Entrega para a Pessoa 2 — comparação das arquiteturas

**Pergunta:** qual arquitetura tem maior acurácia no pior grupo com ERM, e em qual delas o DFR mais melhora essa métrica?

A Pessoa 1 roda a ResNet-50. As outras quatro arquiteturas já estão disponíveis no [mesmo adaptador](../reproduction/architectures.py). Não é preciso reimplementar o DFR, as divisões ou as métricas.

## Execução

1. Configure o ambiente e os dados conforme o [protocolo](REPRODUCAO_WATERBIRDS.md). Rode `python -m reproduction check`.
2. Execute as quatro arquiteturas. Cada comando gera ERM e DFR para as cinco sementes e pode ser repetido caso a sessão seja interrompida:

   ```sh
   python -m reproduction run --architecture densenet121
   python -m reproduction run --architecture efficientnet_b0
   python -m reproduction run --architecture shufflenet_v2_x0_5
   python -m reproduction run --architecture mobilenet_v3_small
   ```

   Para confirmar o pipeline antes de comprometer as cinco sementes, rode uma
   semente de cada rede e verifique se os dois estágios terminam:

   ```sh
   python -m reproduction run --architecture shufflenet_v2_x0_5 --seed 1
   python -m reproduction run --architecture mobilenet_v3_small --seed 1
   ```

   Se já existir uma execução de `efficientnet_b0` com este mesmo protocolo
   (mesmas divisões, 100 épocas, sementes 1–5 e pesos ImageNet-1K), reaproveite
   a pasta `runs/efficientnet_b0/` em vez de repetir o treino. Confira o
   `protocol.json` antes de misturar os resultados.

3. Receba da Pessoa 1 os arquivos `protocol.json`, `results.csv`, `summary.csv` e `paper_comparison.csv` de `runs/resnet50/`. Confirme o hash do `metadata.csv`, as sementes e a configuração; guarde os logs e checkpoints à parte. Use os mesmos lotes, resolução, divisões e orçamento de ajuste. Registre diferenças de hardware ou ambiente. Se faltar memória em alguma arquitetura, use uma GPU com mais memória ou combine e documente um novo protocolo para todos os modelos antes de comparar.

## Como comparar

Cada arquitetura gera dez linhas em `results.csv`: dois métodos para cada uma das cinco sementes. Use o `summary.csv` para apresentar média e desvio padrão por método. Para medir o efeito do DFR, calcule por semente `pior grupo DFR − pior grupo ERM` em pontos percentuais e reporte a média dessas diferenças. Indique também qual grupo é o pior em cada método e se houve troca entre desempenho no pior grupo e acurácia ponderada.

Depois das cinco sementes de todas as arquiteturas, os dois scripts abaixo
consolidam a comparação e geram a figura:

```sh
python scripts/aggregate_results.py
python scripts/plot_person2_results.py
```

`aggregate_results.py` grava `summary_all.csv`, `deltas.csv`,
`deltas_summary.csv` e `resumo.md` em `runs/aggregate/`. Ele ignora, com aviso,
arquiteturas que não tenham ERM e DFR nas cinco sementes. A figura usa os mesmos
dados e mostra o pior grupo e a média ponderada lado a lado.

As limitações que valem para qualquer conclusão estão em
[Limitações das abordagens implementadas](LIMITACOES_ABORDAGENS.md). Leia antes
de comparar arquiteturas: a receita é comum, o DFR usa rótulos de grupo e o
pior grupo tem poucos exemplos.

O CSV distingue `weighted_accuracy_pct` (média com pesos dos grupos no treino) de `micro_accuracy_pct` (acurácia geral no teste). `group_gap_pp` é maior menos menor acurácia entre os quatro grupos. Não junte números exploratórios de `results/` com dados novos de `runs/`.

A análise deve reconhecer que o DFR usa rótulos de grupo na validação, que uma única base não garante generalização e que os pesos pré-treinados e receitas de pré-treino podem diferir mesmo usando ImageNet-1K. A receita de ajuste comum favorece uma comparação sob as mesmas condições; ela não mede o melhor desempenho possível de cada arquitetura. Grad-CAM pode complementar a análise de redes convolucionais, mas não é uma métrica de comparação nem prova que o modelo usa o fundo.
