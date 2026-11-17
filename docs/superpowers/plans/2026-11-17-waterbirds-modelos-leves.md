# Modelos leves para a Pessoa 2 — plano de implementação

**Objetivo:** manter a ResNet-50 da Pessoa 1 como referência e viabilizar a comparação entre cinco arquiteturas no Waterbirds. As quatro alternativas leves usam os mesmos dados, ERM, DFR e métricas; o notebook da Pessoa 1 não muda.

1. Testar se o runner aceita os quatro nomes e se as redes produzem duas classes, permitem treinar a camada de saída e expõem a entrada correta para o DFR.
2. Registrar ShuffleNet V2 ×0,5, MobileNetV3-Small e MNASNet 0.5; manter EfficientNet-B0 para preservar trabalho sob o protocolo anterior.
3. Preparar notebook independente com uma semente por padrão e cinco na execução final. Exportar os resultados com os mesmos nomes; documentar o reaproveitamento de EfficientNet-B0.
4. Conferir o notebook, o checkout dos autores e os testes com PyTorch; verificar que o notebook da Pessoa 1 não foi alterado.
