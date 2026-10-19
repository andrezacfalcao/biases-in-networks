# Viés em Redes Neurais: Redução de Atalhos e Correlações Espúrias no Dataset Waterbirds

Repositório com a implementação completa, notebook para Google Colab, apresentação em slides (.pptx) e resultados experimentais reais para o seminário acadêmico de Redes Neurais.

---

## 📁 Estrutura do Projeto

```
biases-in-networks/
│
├── waterbirds_bias_reduction.ipynb      # Notebook Jupyter / Google Colab (pronto para rodar em GPU T4)
├── seminario_vies_redes_neurais.pptx    # Apresentação completa em slides (13 slides, notas de orador, 3 apresentadores)
│
├── results/
│   ├── waterbirds_model_comparison.csv       # Tabela de resultados reais (Avg Acc, WGA, Disparidade, 4 Grupos)
│   ├── waterbirds_accuracy_comparison.png    # Gráfico de barras: Acurácia Média vs. Pior Grupo (WGA)
│   ├── waterbirds_group_accuracies.png       # Gráfico detalhado de acurácia por grupo de teste
│   ├── waterbirds_gradcam_comparison.png     # Mapas de atenção Grad-CAM (ERM vs. DFR)
│   ├── waterbirds_gradcam_dual_minority.png  # Comparativo duplo de casos minoritários
│   ├── waterbirds_groups_examples.png        # Exemplos visuais dos 4 grupos
│   └── waterbirds_split_distribution.png     # Gráfico da distribuição de treino, validação e teste
│
├── scripts/
│   ├── run_experiments.py              # Script de extração de features e treinamento dos modelos
│   ├── generate_gradcam.py             # Script de geração dos mapas de saliência Grad-CAM
│   ├── generate_dataset_overview.py    # Visualizações do dataset
│   ├── create_notebook.py              # Gerador do notebook Colab
│   └── create_presentation.py          # Gerador da apresentação .pptx
│
└── data/
    └── waterbirds_v1.0/                # Dataset oficial Waterbirds (metadados e imagens)
```

---

## 📊 Resultados Empíricos Obtidos no Teste (5.794 Imagens)

Espinha dorsal: **ResNet-50** pré-treinada na ImageNet-1K ($d = 2048$).

| Método | Acurácia Média (Avg Acc) | Pior Grupo (Worst-Group Acc - WGA) | Disparidade (Max - Min) | G0: Terra/Terra (Maioria) | G1: Terra/Água (Minoria) | G2: Água/Terra (Minoria Crítica) | G3: Água/Água (Maioria) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ERM (Padrão)** | 87.88% | **67.45%** | **32.11%** | 99.56% | 80.31% | **67.45%** | 93.93% |
| **Re-weighting (Treino)** | 89.77% | **74.30%** | **24.02%** | 98.31% | 84.66% | **74.30%** | 93.15% |
| **DFR (Kirichenko et al.)** | **90.82%** | **89.56%** | **3.03%** | 92.59% | 89.62% | **89.56%** | 90.03% |
| **JTT (Liu et al.)** | 87.50% | **66.36%** | **32.89%** | 99.25% | 79.96% | **66.36%** | 93.93% |

---

## 🚀 Como Executar

### Opção 1: Google Colab
1. Abra o arquivo [`waterbirds_bias_reduction.ipynb`](waterbirds_bias_reduction.ipynb) no Google Colab.
2. Em *Ambiente de Execução* -> *Alterar tipo de ambiente de execução*, selecione **GPU T4**.
3. Execute todas as células em sequência. O download e treinamento levarão de 3 a 5 minutos.

### Opção 2: Localmente
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # ou pip install torch torchvision wilds scikit-learn matplotlib seaborn pillow python-pptx
python scripts/run_experiments.py
python scripts/generate_gradcam.py
```

---

## 📚 Referências Auditadas
1. **Sagawa et al. (ICLR 2020)**: *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization.*
2. **Kirichenko et al. (NeurIPS 2022)**: *Last Layer Re-Training is Sufficient for Robustness (DFR).*
3. **Geirhos et al. (Nature Machine Intelligence 2020)**: *Shortcut learning in deep neural networks.*
4. **Liu et al. (ICML 2021)**: *Just Train Twice: Improving Group Robustness without Attribute Annotations (JTT).*
5. **Selvaraju et al. (ICCV 2017)**: *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
