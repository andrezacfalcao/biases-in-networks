# Plano: reprodução Waterbirds e entrega para a Pessoa 2

Escopo: duas linhas da Tabela 2 de Kirichenko et al. (ResNet-50, ERM e DFR no Waterbirds) e um pipeline reutilizável para comparar as outras arquiteturas. O notebook e os resultados antigos continuam disponíveis como piloto.

1. Fixar a revisão pública dos autores. Preparar uma cópia descartável com um patch auditável: última época fixa; teste fora do treino; dez ajustes DFR; semente registrada e avaliação serializada em JSON. Verificar que o patch aplica ao commit real e que os scripts compilam.
2. Criar um adaptador para trocar o classificador de cada arquitetura e extrair a entrada dessa camada. Preservar, para ResNet-50, os pesos supervisionados V1 usados pelos autores. Verificar as dimensões em CPU.
3. Fixar as divisões oficiais, métricas e esquema do CSV. Testar uma média manual, um grupo ausente, metadados inválidos e a conversão de resultados por semente; exigir as cinco execuções para o resumo de referência.
4. Expor um comando de verificação sem treino e um para CUDA. Documentar configuração, comandos, divergências do upstream, entrega para a Pessoa 2 e o que depende de executar as cinco sementes.
