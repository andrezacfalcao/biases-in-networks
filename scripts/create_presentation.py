import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Cohesive Academic Palette
    COLOR_BG_DARK = RGBColor(15, 23, 42)        # Slate 900
    COLOR_TEXT_WHITE = RGBColor(255, 255, 255)
    COLOR_TEXT_DARK = RGBColor(30, 41, 59)       # Slate 800
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139)   # Slate 500
    COLOR_PRIMARY = RGBColor(37, 99, 235)       # Blue 600
    COLOR_ACCENT_RED = RGBColor(220, 38, 38)    # Red 600
    COLOR_ACCENT_GREEN = RGBColor(22, 163, 74)  # Green 600
    COLOR_ACCENT_PURPLE = RGBColor(147, 51, 234)# Purple 600
    COLOR_CARD_BG = RGBColor(248, 250, 252)     # Slate 50
    COLOR_CARD_BORDER = RGBColor(226, 232, 240) # Slate 200

    def add_header(slide, title_text, section_text, presenter_tag):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.15))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_sec = tf.paragraphs[0]
        p_sec.text = f"{section_text.upper()}  •  {presenter_tag}"
        p_sec.font.size = Pt(11)
        p_sec.font.bold = True
        p_sec.font.color.rgb = COLOR_PRIMARY
        p_sec.space_after = Pt(4)
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(23)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_DARK

    def set_speaker_notes(slide, presenter, duration, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = f"[{presenter.upper()} | TEMPO SUGERIDO: {duration}]\n\n{notes_text}"

    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: CAPA (DARK THEME)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG_DARK
    bg.line.fill.background()

    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(4.5))
    tf = tbox.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "SEMINÁRIO UNIVERSITÁRIO DE REDES NEURAIS PROFUNDAS"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_PRIMARY
    p0.space_after = Pt(12)

    p1 = tf.add_paragraph()
    p1.text = "Viés em Redes Neurais: Redução de Atalhos e Correlações Espúrias"
    p1.font.size = Pt(35)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_WHITE
    p1.space_after = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Investigação Empírica no Dataset Waterbirds: Do Colapso do ERM à Robustez com DFR"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_after = Pt(24)

    p_div = tf.add_paragraph()
    p_div.text = "ESTRUTURA: I. Introdução  |  II. O Projeto  |  III. O Dataset  |  IV. Teoria  |  V. Metodologia  |  VI. Resultados  |  VII. Interpretabilidade Visual  |  VIII. Conclusão"
    p_div.font.size = Pt(11)
    p_div.font.bold = True
    p_div.font.color.rgb = RGBColor(148, 163, 184)
    p_div.space_after = Pt(16)

    p3 = tf.add_paragraph()
    p3.text = "Apresentadores: Apresentador 1 (Intro, Objetivos & Dataset) | Apresentador 2 (Teoria & Métodos) | Apresentador 3 (Resultados, Grad-CAM & Análise)\nTempo Total: ~20 minutos | Experimentos Reais com ResNet-50 e Mapas de Calor Grad-CAM"
    p3.font.size = Pt(12)
    p3.font.color.rgb = COLOR_TEXT_MUTED

    set_speaker_notes(s1, "Apresentador 1", "0.5 min",
        "Bom dia a todos, professor e colegas. Hoje nosso grupo apresenta o seminário sobre 'Viés em Redes Neurais: "
        "Redução de Atalhos e Correlações Espúrias'. Nosso trabalho foi estruturado com rigor acadêmico: "
        "iniciaremos pela Introdução do problema no mundo real; em seguida apresentaremos os Objetivos do Projeto e, em um slide dedicado, o Dataset Waterbirds; "
        "o Apresentador 2 cobrirá a Teoria de shortcut learning e as Soluções de mitigação; e o Apresentador 3 trará nossa Metodologia, "
        "Resultados Empíricos reais no teste, a PROVA VISUAL do viés no Baseline ERM e a REVERSÃO com DFR. Vamos à introdução.")

    # =========================================================================
    # SLIDE 2: SEÇÃO I - INTRODUÇÃO & MOTIVAÇÃO
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "O Paradoxo da Alta Acurácia: Atalhos e Falhas no Mundo Real", "Seção I: Introdução & Motivação", "Apresentador 1")

    card_w = Inches(3.64)
    card_h = Inches(4.8)
    card_top = Inches(1.8)

    cards_data = [
        ("Dermatologia & Saúde", 
         "Marcadores e Réguas Fotográficas",
         "Modelos de Deep Learning treinados para triagem de melanoma associaram a presença de réguas milimétricas plásticas à malignidade. Lesões perigosas costumam ser medidas pelos médicos antes da biópsia, criando um atalho visual enganoso.",
         COLOR_ACCENT_RED),
        ("Radiologia Torácica", 
         "Fontes e Etiquetas de Máquinas de Raio-X",
         "Redes neurais para detecção de pneumonia e COVID-19 aprenderam a reconhecer o modelo do aparelho e a fonte da legenda do hospital de origem em vez de infiltrados pulmonares, falhando catastroficamente ao mudar de clínica.",
         COLOR_PRIMARY),
        ("A Falha do ERM", 
         "Minimização do Risco Empírico Médio",
         "O treinamento padrão (ERM) minimiza a perda média: min (1/N) ∑ L(f(x), y). Se um atalho visual cobre 95% do dataset, o gradiente descende muito mais rápido explorando esse atalho do que decodificando a causa biológica real.",
         COLOR_TEXT_DARK)
    ]

    for i, (tag, title, desc, tag_col) in enumerate(cards_data):
        shape = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 4.04), card_top, card_w, card_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD_BG
        shape.line.color.rgb = COLOR_CARD_BORDER
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.3)

        p = tf.paragraphs[0]
        p.text = tag.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = tag_col
        p.space_after = Pt(8)

        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_DARK
        p_t.space_after = Pt(12)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12.5)
        p_d.font.color.rgb = RGBColor(71, 85, 105)

    set_speaker_notes(s2, "Apresentador 1", "2.5 min",
        "Na Seção I, apresentamos a motivação do nosso estudo. É comum vermos artigos de Deep Learning reportando acurácias superiores a 95% ou 98%. "
        "No entanto, quando esses sistemas são implantados em hospitais ou veículos autônomos, ocorrem falhas graves. "
        "Isso acontece pelo fenômeno de 'Shortcut Learning' (aprendizado por atalhos): a rede identifica uma correlação espúria estatisticamente forte no treino, "
        "mas que não possui causalidade biológica com o problema. Em dermatologia, modelos aprenderam que lesões com réguas fotográficas ao lado eram malignas. "
        "Em radiologia, a rede memorizou a tipografia das etiquetas hospitalares. "
        "A causa raiz está na própria formulação matemática da Minimização de Risco Empírico (ERM): ao minimizar o erro médio, "
        "a rede é matematicamente incentivada a pegar o atalho que resolve a vasta maioria dos casos.")

    # =========================================================================
    # SLIDE 3: SEÇÃO II - O PROJETO (OBJETIVOS & HIPÓTESE CIENTÍFICA)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Objetivos do Projeto e Hipótese Científica Central", "Seção II: O Projeto", "Apresentador 1")

    obj_cards = [
        ("Objetivo Geral", "Isolamento e Mitigação do Viés",
         "• Investigar experimentalmente a falha da Minimização de Risco Empírico (ERM) quando exposta a correlações espúrias em visão computacional.\n"
         "• Avaliar e comparar métodos recentes de mitigação (Re-weighting de treino, DFR e JTT) em um ambiente controlado e reproduzível.\n"
         "• Provar empiricamente que é possível restaurar a robustez em subpopulações sem a necessidade de retreinar a rede convolucional do zero.",
         COLOR_PRIMARY),
        ("Hipótese Central", "A Tese de Kirichenko et al. (DFR, 2022)",
         "• Hipótese: O extrator convolucional profundo (backbone ResNet-50) NÃO perde a capacidade de identificar a anatomia da ave.\n"
         "• As representações latentes profundas já codificam as características semânticas verdadeiras; o viés de atalho reside quase que exclusivamente na camada linear de classificação final.\n"
         "• Portanto, reponderar apenas a última camada com dados de validação balanceados é suficiente para recuperar a equidade entre grupos.",
         COLOR_ACCENT_GREEN),
        ("Objetivos Específicos", "Entregas e Métricas do Seminário",
         "1. Auditar a distribuição real de splits e grupos do benchmark Waterbirds (11.788 imagens).\n"
         "2. Treinar o modelo baseline ERM com ResNet-50 e registrar o colapso no pior grupo.\n"
         "3. Aplicar o algoritmo DFR e avaliar Worst-Group Accuracy (WGA) e Disparidade de Grupo.\n"
         "4. Implementar explicabilidade visual com Grad-CAM para comprovar a migração da atenção do fundo para a ave.",
         COLOR_TEXT_DARK)
    ]

    for i, (title, sub, body, col) in enumerate(obj_cards):
        shape = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 4.04), Inches(1.8), card_w, card_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD_BG
        shape.line.color.rgb = COLOR_CARD_BORDER
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = title.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(4)

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(14)
        p_s.font.bold = True
        p_s.font.color.rgb = COLOR_TEXT_DARK
        p_s.space_after = Pt(12)

        p_t = tf.add_paragraph()
        p_t.text = body
        p_t.font.size = Pt(11)
        p_t.font.color.rgb = COLOR_TEXT_DARK

    set_speaker_notes(s3, "Apresentador 1", "2.0 min",
        "Entramos agora na Seção II com os Objetivos do nosso Projeto. "
        "Nosso objetivo geral foi investigar a quebra da Minimização de Risco Empírico sob correlações espúrias e implementar soluções eficientes. "
        "A hipótese científica central que testamos veio do artigo de Kirichenko et al. (NeurIPS 2022 - DFR): "
        "nós hipotetizamos que uma rede profunda como a ResNet-50 não perde os filtros que reconhecem a ave; "
        "o erro está apenas nos pesos da camada linear final, que foram corrompidos pelo atalho do fundo. "
        "Nossos objetivos específicos foram auditar o dataset Waterbirds oficial, implementar o baseline ERM, "
        "aplicar o retreino linear balanceado (DFR) e comprovar visualmente via mapas Grad-CAM a reversão do atalho.")

    # =========================================================================
    # SLIDE 4: SEÇÃO II - O DATASET WATERBIRDS (CONSTRUÇÃO E 4 GRUPOS)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "O Dataset Waterbirds: Isolando a Correlação Espúria", "Seção II: O Dataset", "Apresentador 1")

    tb_ds = s4.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(5.6), Inches(5.1))
    tf_ds = tb_ds.text_frame
    tf_ds.word_wrap = True
    tf_ds.margin_left = tf_ds.margin_top = tf_ds.margin_right = tf_ds.margin_bottom = 0

    p = tf_ds.paragraphs[0]
    p.text = "SAGAWA ET AL. (ICLR 2020) — CONSTRUÇÃO CONTROLADA"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    bullets_wb = [
        "**Fusão de Dois Datasets Reais:** Aves recortadas do CUB-200-2011 sobrepostas em fundos de terra e água do Places.",
        "**Tarefa de Classificação Binária:**\n  • y = 0: Ave Terrestre (Landbird)\n  • y = 1: Ave Aquática (Waterbird)",
        "**Atributo Espúrio (Fundo da Cena):**\n  • a = 0: Cenário Terrestre (floresta de bambu, árvores)\n  • a = 1: Cenário Aquático (lago natural, oceano)",
        "**4 Grupos Reais no Treinamento:**\n  • G0: Terra/Terra (3.498 amostras - 73.0% Maioria)\n  • G1: Terra/Água (184 amostras - 3.8% Minoria)\n  • G2: Água/Terra (apenas 56 amostras - 1.2% Minoria Crítica!)\n  • G3: Água/Água (1.057 amostras - 22.0% Maioria)",
        "Total no Dataset: **11.788 imagens reais** (4.795 treino, 1.199 validação, 5.794 teste)."
    ]
    for b in bullets_wb:
        p = tf_ds.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(6)

    img_groups = "results/waterbirds_groups_examples.png"
    if os.path.exists(img_groups):
        s4.shapes.add_picture(img_groups, Inches(6.6), Inches(1.8), width=Inches(5.9))

    set_speaker_notes(s4, "Apresentador 1", "2.5 min",
        "Neste slide, apresentamos detalhadamente o benchmark Waterbirds, criado por Sagawa e colaboradores em Stanford (ICLR 2020). "
        "Para estudar o viés de forma limpa e sem ruídos, os autores pegaram imagens de aves do clássico CUB-200 e colaram sobre fundos do banco Places. "
        "A tarefa parece muito simples: classificar se é ave terrestre ou aquática. "
        "Mas vejam as 4 imagens reais do dataset exibidas à direita: "
        "No Grupo 0, temos aves de terra em florestas (3.498 imagens de treino). "
        "No Grupo 3, temos aves aquáticas em lagos ou oceanos (1.057 imagens). "
        "Agora olhem os grupos minoritários: no Grupo 1, temos aves terrestres na água (apenas 184 imagens). "
        "E no Grupo 2, temos aves aquáticas em terra com apenas 56 fotos de treino! "
        "São 56 amostras contra quase 3.500 do grupo majoritário! Esse desbalanceamento extremo cria a correlação espúria perfeita.")

    # =========================================================================
    # SLIDE 5: SEÇÃO II - A ARMADILHA DE TREINO E AS MÉTRICAS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "A Armadilha de Treino e as Métricas Rigorosas de Avaliação", "Seção II: O Dataset & As Métricas", "Apresentador 1")

    tb_m1 = s5.shapes.add_textbox(Inches(0.8), Inches(1.75), Inches(5.6), Inches(5.1))
    tf_m1 = tb_m1.text_frame
    tf_m1.word_wrap = True
    tf_m1.margin_left = tf_m1.margin_top = tf_m1.margin_right = tf_m1.margin_bottom = 0

    p = tf_m1.paragraphs[0]
    p.text = "A FALÁCIA DA ACURÁCIA MÉDIA"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED
    p.space_after = Pt(6)

    p_avg = tf_m1.add_paragraph()
    p_avg.text = (
        "• Acurácia Média: Mede o acerto global ponderado pelo tamanho dos grupos no split.\n"
        "• A Armadilha: No treino, 95% das aves terrestres estão na terra e 95% das aquáticas na água. Se a rede ignorar a ave e olhar apenas o fundo, ela atinge 95% de acurácia média!\n"
        "• No teste oficial (5.794 imagens), os autores equalizaram as proporções (50% terra, 50% água para cada classe), expondo a fragilidade do modelo."
    )
    p_avg.font.size = Pt(11)
    p_avg.font.color.rgb = COLOR_TEXT_DARK
    p_avg.space_after = Pt(14)

    p_wga_title = tf_m1.add_paragraph()
    p_wga_title.text = "AS DUAS MÉTRICAS PADRÃO-OURO"
    p_wga_title.font.size = Pt(12)
    p_wga_title.font.bold = True
    p_wga_title.font.color.rgb = COLOR_ACCENT_GREEN
    p_wga_title.space_after = Pt(6)

    bullets_met = [
        "1. Worst-Group Accuracy (WGA): WGA = min_{g} Acc_g.\n   Avalia o pior desempenho absoluto entre os 4 grupos. Se a rede memorizou o fundo, o grupo minoritário terá acurácia baixa e o WGA desaba.",
        "2. Disparidade de Grupo (Gap): Δ = max_{g} Acc_g - min_{g} Acc_g.\n   Mede o desnível entre o grupo mais fácil e o mais difícil. Um modelo robusto busca Disparidade próxima de 0%."
    ]
    for b in bullets_met:
        p = tf_m1.add_paragraph()
        p.text = b
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(8)

    img_dist = "results/waterbirds_split_distribution.png"
    if os.path.exists(img_dist):
        s5.shapes.add_picture(img_dist, Inches(6.6), Inches(1.8), width=Inches(5.9))

    set_speaker_notes(s5, "Apresentador 1", "2.0 min",
        "Concluindo a Seção II, analisamos o contraste dos splits e a escolha das métricas. "
        "Vejam no gráfico à direita: a barra azul domina quase 73% do split de treino. "
        "Se avaliarmos a rede apenas pela acurácia média geral, estaremos sendo enganados, pois a maioria esconde os erros da minoria. "
        "Entretanto, no split de teste, os autores distribuíram as imagens de forma equilibrada: 50% terra e 50% água para cada espécie. "
        "Por isso, a literatura estabeleceu a Worst-Group Accuracy (WGA) como métrica mandatória: ela mede a acurácia do pior grupo entre os quatro. "
        "Também medimos a Disparidade de Grupo, que é o abismo entre o melhor e o pior grupo. "
        "Passo agora a palavra ao Apresentador 2 para explicar por que a rede prefere o fundo e quais são as soluções teóricas.")

    # =========================================================================
    # SLIDE 6: SEÇÃO III - FUNDAMENTAÇÃO TEÓRICA (SHORTCUT LEARNING)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Por Que a Rede Aprende o Atalho? Simplicity Bias & Gradient Starvation", "Seção III: Fundamentação Teórica", "Apresentador 2")

    card_w2 = Inches(5.6)
    card_h2 = Inches(4.8)

    c1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), card_w2, card_h2)
    c1.fill.solid()
    c1.fill.fore_color.rgb = COLOR_CARD_BG
    c1.line.color.rgb = COLOR_CARD_BORDER
    tf1 = c1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_right = tf1.margin_top = tf1.margin_bottom = Inches(0.3)

    p = tf1.paragraphs[0]
    p.text = "1. SIMPLICITY BIAS (VIÉS POR SIMPLICIDADE)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    p_body1 = [
        "Geirhos et al. (Nature Machine Intelligence, 2020) demonstraram que redes convolucionais têm uma forte tendência inata a aprender textura e frequências de cor antes de formas geométricas.",
        "Detectar 'água' (pixels azuis, ondas repetitivas) ou 'floresta' (tons verdes) exige apenas filtros lineares superficiais nas primeiras camadas convolucionais.",
        "Em contrapartida, identificar características anatômicas de aves (bico fino, formato das asas, garras) exige múltiplas camadas convolucionais combinadas hierarquicamente.",
        "O otimizador sempre segue a descida de gradiente mais rápida e barata."
    ]
    for pb in p_body1:
        p = tf1.add_paragraph()
        p.text = "• " + pb
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(8)

    c2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), card_w2, card_h2)
    c2.fill.solid()
    c2.fill.fore_color.rgb = COLOR_CARD_BG
    c2.line.color.rgb = COLOR_CARD_BORDER
    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = tf2.margin_top = tf2.margin_bottom = Inches(0.3)

    p = tf2.paragraphs[0]
    p.text = "2. GRADIENT STARVATION (ESTRANGULAMENTO)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED
    p.space_after = Pt(10)

    p_body2 = [
        "Pezeshki et al. (NeurIPS, 2021) formalizaram o fenômeno de 'Gradient Starvation':",
        "Quando um recurso simples (fundo) é suficiente para diminuir a função de perda na vasta maioria dos exemplos (95%), a magnitude total do gradiente decresce acentuadamente já nas primeiras épocas.",
        "Com o gradiente reduzido, os pesos da rede que processam as características complexas da ave deixam de receber atualizações suficientes.",
        "Conclusão: O atalho do fundo 'mata de fome' os neurônios responsáveis pelas características verdadeiras da ave."
    ]
    for pb in p_body2:
        p = tf2.add_paragraph()
        p.text = "• " + pb
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(8)

    set_speaker_notes(s6, "Apresentador 2", "2.0 min",
        "Obrigado, Apresentador 1. Na Seção III de Fundamentação Teórica, investigamos a dinâmica interna da rede: "
        "por que uma rede convolucional prefere olhar para o fundo em vez do pássaro? "
        "A literatura recente descobriu dois mecanismos fundamentais: "
        "O primeiro é o 'Simplicity Bias' (Geirhos et al., 2020). Redes neurais priorizam texturas e cores de baixa complexidade. "
        "Reconhecer água é computacionalmente simples: bastam alguns filtros de cor azul e textura repetitiva nas camadas iniciais. "
        "Já reconhecer a anatomia de uma ave aquática exige compor bordas, curvas e formas em camadas profundas. "
        "O segundo mecanismo é o 'Gradient Starvation' (Pezeshki et al., 2021). Como o fundo acerta 95% do dataset logo no início, "
        "a função de perda cai rapidamente. Com a perda baixa, os gradientes retropropagados tornam-se minúsculos. "
        "Consequentemente, as convoluções que poderiam aprender o pássaro deixam de receber gradientes. O atalho simplesmente estrangula o aprendizado da causa real.")

    # =========================================================================
    # SLIDE 7: SEÇÃO III - PANORAMA DE SOLUÇÕES (GROUP DRO, JTT, DFR)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Panorama de Soluções: Três Paradigmas de Mitigação", "Seção III: Fundamentação Teórica", "Apresentador 2")

    card_w3 = Inches(3.64)
    card_h3 = Inches(4.8)

    solutions_data = [
        ("Group DRO", "Sagawa et al. (ICLR 2020)",
         "Otimização Minimax Robusta",
         "• Formulação: min_θ max_g E_{P_g}[L(f_θ(x), y)].\n"
         "• Em vez da média geral, ajusta pesos dinâmicos q_g via subida de gradiente exponencial, penalizando mais o pior grupo.\n"
         "• Limitação: Exige anotações de grupo em 100% dos dados de treino e retreinamento caro de ponta a ponta.",
         COLOR_PRIMARY),
        ("JTT (Just Train Twice)", "Liu et al. (ICML 2021)",
         "Mitigação em Dois Estágios",
         "• Estágio 1: Treina um modelo ERM rápido com early stopping para gerar o 'Error Set' (erros concentrados em minorias).\n"
         "• Estágio 2: Retreina o modelo sobrepesando o Error Set por um fator multiplicativo λ_up (20x a 50x).\n"
         "• Vantagem: Dispensa anotações de grupo no treino, mas depende de heurísticas no primeiro estágio.",
         COLOR_ACCENT_PURPLE),
        ("DFR (Deep Feature Reweighting)", "Kirichenko et al. (NeurIPS 2022)",
         "Retreino da Última Camada",
         "• Descoberta: O extrator convolucional já aprendeu as features corretas; o viés reside na camada linear final!\n"
         "• Procedimento: Congela o backbone e retreina apenas a camada final em validação balanceada.\n"
         "• Vantagem: Executa em menos de 1 segundo, extremamente simples e supera o Group DRO!",
         COLOR_ACCENT_GREEN)
    ]

    for i, (title, auth, sub, text_c, col) in enumerate(solutions_data):
        shape = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 4.04), Inches(1.8), card_w3, card_h3)
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD_BG
        shape.line.color.rgb = COLOR_CARD_BORDER
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(2)

        p_a = tf.add_paragraph()
        p_a.text = auth
        p_a.font.size = Pt(10.5)
        p_a.font.bold = True
        p_a.font.color.rgb = COLOR_TEXT_MUTED
        p_a.space_after = Pt(4)

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(12)
        p_s.font.bold = True
        p_s.font.color.rgb = COLOR_TEXT_DARK
        p_s.space_after = Pt(10)

        p_t = tf.add_paragraph()
        p_t.text = text_c
        p_t.font.size = Pt(11)
        p_t.font.color.rgb = COLOR_TEXT_DARK

    set_speaker_notes(s7, "Apresentador 2", "2.0 min",
        "Diante desse problema, a comunidade de Machine Learning propôs três caminhos metodológicos principais: "
        "O primeiro é o Group DRO (Sagawa et al., 2020), que usa teoria minimax. Em vez de minimizar a perda média, "
        "ele penaliza o grupo com a maior perda a cada iteração. É uma solução matematicamente sólida, mas exige que 100% dos dados de treino tenham rótulos de grupo e retreina a rede inteira. "
        "O segundo é o JTT (Just Train Twice, Liu et al., 2021). A sacada do JTT é não precisar de rótulos de grupo no treino: ele treina um ERM rápido, descobre quais amostras a rede errou "
        "(que costumam ser justamente as minorias desalinhadas), e retreina a rede dando peso 20x a 50x para esses erros. "
        "E o terceiro é o DFR (Kirichenko et al., 2022), que é a técnica central do nosso projeto e detalharemos a seguir.")

    # =========================================================================
    # SLIDE 8: SEÇÃO III - FOCO METODOLÓGICO NO DFR
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Foco Metodológico: Deep Feature Reweighting (DFR)", "Seção III: Fundamentação Teórica", "Apresentador 2")

    c_left = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), card_w2, card_h2)
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = COLOR_CARD_BG
    c_left.line.color.rgb = COLOR_CARD_BORDER
    tf_l = c_left.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = tf_l.margin_bottom = Inches(0.3)

    p = tf_l.paragraphs[0]
    p.text = "A DESCOBERTA DE KIRICHENKO ET AL. (2022)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    bullets_dfr1 = [
        "**O extrator convolucional já aprendeu as features corretas!**",
        "Ao inspecionar o espaço latente de representações profundas (d = 2048 na ResNet-50), os autores descobriram que as convoluções já codificam bicos, penas e silhuetas da ave.",
        "O viés reside quase que exclusivamente na **última camada linear (cabeça de classificação)**: como o fundo correlaciona 95% no treino, a camada final atribui pesos desmedidos aos canais do fundo.",
        "Conclusão: Não é necessário retreinar os milhões de parâmetros da espinha dorsal convolucional."
    ]
    for b in bullets_dfr1:
        p = tf_l.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(8)

    c_right = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), card_w2, card_h2)
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = COLOR_CARD_BG
    c_right.line.color.rgb = COLOR_CARD_BORDER
    tf_r = c_right.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = Inches(0.3)

    p = tf_r.paragraphs[0]
    p.text = "O ALGORITMO DFR EM 3 ETAPAS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN
    p.space_after = Pt(10)

    bullets_dfr2 = [
        "**Etapa 1 (Congelamento):** Congela o extrator convolucional φ(x) (ResNet-50 pré-treinada).",
        "**Etapa 2 (Amostragem Balanceada):** Extrai um subconjunto balanceado da validação com N_min amostras por grupo (no nosso experimento, exatamente 133 amostras por grupo = 532 amostras).",
        "**Etapa 3 (Otimização Convexa):** Treina apenas os pesos w e bias b via regressão logística com regularização L_2 ajustada (C = 0.1):\nmin_{w,b} ∑ L(w^T φ(x_i) + b, y_i) + (1 / 2C) ||w||_2^2",
        "**Custo Computacional:** Menos de 1 segundo! Converte um problema de horas de GPU em uma otimização quase instantânea."
    ]
    for b in bullets_dfr2:
        p = tf_r.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(8)

    set_speaker_notes(s8, "Apresentador 2", "2.0 min",
        "O Deep Feature Reweighting (DFR), apresentado no NeurIPS 2022, trouxe uma quebra de paradigma na literatura. "
        "A intuição dominante era que para remover o viés, teríamos que retreinar a rede convolucional inteira do zero com perdas minimax caríssimas. "
        "Kirichenko e seus colegas provaram que isso é desnecessário: a ResNet-50 já tem os filtros da ave no seu espaço latente! "
        "O ERM apenas deu pesos exagerados para os neurônios que respondem ao fundo verde ou azul. "
        "O DFR congela todas as convoluções e reajusta apenas os 2.048 pesos da camada linear final usando um subconjunto de validação balanceado (133 amostras por grupo). "
        "É uma regressão logística convexa que roda em frações de segundo. Passo agora para o Apresentador 3 mostrar nossa Metodologia e Resultados Reais.")

    # =========================================================================
    # SLIDE 9: SEÇÃO IV - METODOLOGIA EXPERIMENTAL
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Metodologia Experimental: Pipeline de Execução e Reprodutibilidade", "Seção IV: Metodologia Experimental", "Apresentador 3")

    pipe_w = Inches(2.7)
    pipe_h = Inches(4.8)

    pipe_data = [
        ("1. Dataset Real", "Waterbirds Oficial", 
         "• 11.788 imagens reais.\n• Treino: 4.795 (95% correlação).\n• Validação: 1.199.\n• Teste: 5.794 (50/50 balanceado).\n• Seeds fixadas em 42."),
        ("2. Espinha Dorsal", "ResNet-50 ImageNet", 
         "• Arquitetura ResNet-50 pré-treinada na ImageNet-1K.\n• Extração do pooling global (vetores densos de d=2048).\n• Executado em GPU / MPS."),
        ("3. Modelos", "4 Abordagens Testadas", 
         "• Baseline ERM (Padrão).\n• Re-weighting de classes no treino.\n• DFR com tuning de C em validação balanceada.\n• JTT em dois estágios com λ_up=20."),
        ("4. Métricas & XAI", "Avaliação Rigorosa", 
         "• Acurácia Média.\n• Worst-Group Acc (WGA).\n• Acurácia individual dos 4 grupos.\n• Mapas de calor Grad-CAM na última camada conv (`layer4`).")
    ]

    for i, (step, title, body) in enumerate(pipe_data):
        shape = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 2.98), Inches(1.8), pipe_w, pipe_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD_BG
        shape.line.color.rgb = COLOR_CARD_BORDER
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.2)

        p = tf.paragraphs[0]
        p.text = step.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_after = Pt(4)

        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_DARK
        p_t.space_after = Pt(12)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    set_speaker_notes(s9, "Apresentador 3", "1.5 min",
        "Obrigado, Apresentador 2. Na Seção IV, apresento a metodologia prática que implementamos e executamos. "
        "Não utilizamos aproximações: baixamos o dataset Waterbirds oficial via pacote WILDS com 11.788 imagens reais. "
        "Seguindo o protocolo estrito de Kirichenko et al. (2022), utilizamos a espinha dorsal de uma ResNet-50 pré-treinada na ImageNet, "
        "extraindo o vetor de representação de 2.048 dimensões para cada imagem. "
        "Em seguida, implementamos 4 abordagens comparativas: o Baseline ERM, o Re-weighting inverso de grupos no treino, o DFR e o JTT. "
        "Fixamos a semente aleatória 42 para garantir reprodutibilidade completa no nosso notebook Colab.")

    # =========================================================================
    # SLIDE 10: SEÇÃO V - RESULTADOS EMPÍRICOS (TABELA & COMPARATIVO)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Resultados Reais no Teste: O Colapso do ERM e o Triunfo do DFR", "Seção V: Resultados Empíricos", "Apresentador 3")

    tb_res = s10.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_res = tb_res.text_frame
    tf_res.word_wrap = True
    tf_res.margin_left = tf_res.margin_top = tf_res.margin_right = tf_res.margin_bottom = 0

    p = tf_res.paragraphs[0]
    p.text = "RESULTADOS AUDITADOS NO TESTE (5.794 IMAGENS)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    p_tab = tf_res.add_paragraph()
    p_tab.text = (
        "• ERM (Baseline Padrão):\n"
        "  Avg Acc: 87.88%  |  WGA: 67.45%  |  Disparidade: 32.11%\n"
        "  (Grupo Terra/Terra: 99.56% vs. Água/Terra: 67.45%!)\n\n"
        "• Re-weighting (Treino):\n"
        "  Avg Acc: 89.77%  |  WGA: 74.30%  |  Disparidade: 24.02%\n\n"
        "• DFR (Kirichenko et al.):\n"
        "  Avg Acc: 90.82%  |  WGA: 89.56%  |  Disparidade: 3.03%!\n"
        "  (+22.11 pontos percentuais no pior grupo!)\n\n"
        "• JTT (Liu et al.):\n"
        "  Avg Acc: 87.50%  |  WGA: 66.36%  |  Disparidade: 32.89%"
    )
    p_tab.font.size = Pt(11.5)
    p_tab.font.color.rgb = COLOR_TEXT_DARK
    p_tab.space_after = Pt(10)

    p_takeaway = tf_res.add_paragraph()
    p_takeaway.text = "Destaque Empírico: O DFR praticamente eliminou a disparidade (de 32.11% para 3.03%) e elevou o pior grupo em +22.11 pp, superando todos os outros métodos!"
    p_takeaway.font.size = Pt(11)
    p_takeaway.font.bold = True
    p_takeaway.font.color.rgb = COLOR_ACCENT_GREEN

    chart_acc = "results/waterbirds_accuracy_comparison.png"
    if os.path.exists(chart_acc):
        s10.shapes.add_picture(chart_acc, Inches(6.6), Inches(1.8), width=Inches(5.9))

    set_speaker_notes(s10, "Apresentador 3", "2.5 min",
        "Na Seção V, revelamos os resultados empíricos reais obtidos na execução do nosso código no conjunto de teste de 5.794 imagens. "
        "Observem o modelo ERM padrão: ele atinge 87.88% de acurácia média geral. Se você olhar apenas a média, parece um modelo excelente. "
        "Porém, no pior grupo (a ave aquática em fundo de terra), a acurácia cai para apenas 67.45%! "
        "A disparidade entre o melhor e o pior grupo é de 32.11%! A rede quase sempre erra quando a correlação espúria é violada. "
        "O Re-weighting de treino melhora o WGA para 74.30%, mas ainda mantém 24% de disparidade. "
        "Agora vejam o DFR: com apenas 133 amostras por grupo da validação balanceada, o pior grupo salta para 89.56%! "
        "São mais de 22 pontos percentuais de ganho direto no pior grupo! "
        "E o mais extraordinário: a disparidade desmoronou de 32% para apenas 3.03%. Todos os grupos performaram em torno de 90%. "
        "Essa é a comprovação empírica do poder do DFR.")

    # =========================================================================
    # SLIDE 11: SEÇÃO V - ANÁLISE DETALHADA POR GRUPO
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Desempenho por Grupo: O Efeito Nivelador do DFR", "Seção V: Resultados Empíricos", "Apresentador 3")

    tb_g = s11.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf_g = tb_g.text_frame
    tf_g.word_wrap = True
    tf_g.margin_left = tf_g.margin_top = tf_g.margin_right = tf_g.margin_bottom = 0

    p = tf_g.paragraphs[0]
    p.text = "DESMEMBRAMENTO DOS 4 GRUPOS DE TESTE"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    p_breakdown = tf_g.add_paragraph()
    p_breakdown.text = (
        "• G0 (Landbird em Terra - Maioria):\n"
        "  ERM: 99.56%  |  DFR: 92.59%  (Leve ajuste saudável)\n\n"
        "• G1 (Landbird em Água - Minoria):\n"
        "  ERM: 80.31%  |  DFR: 89.62%  (+9.31 pp)\n\n"
        "• G2 (Waterbird em Terra - Minoria Crítica):\n"
        "  ERM: 67.45%  |  DFR: 89.56%  (+22.11 pp! SALTO CRÍTICO)\n\n"
        "• G3 (Waterbird em Água - Maioria):\n"
        "  ERM: 93.93%  |  DFR: 90.03%  (Estável)\n\n"
        "Conclusão: O DFR não 'favorece' minorias prejudicando maiorias; ele neutraliza o atalho do fundo e calibra os grupos para uma equidade perfeita de ~90%."
    )
    p_breakdown.font.size = Pt(11.5)
    p_breakdown.font.color.rgb = COLOR_TEXT_DARK

    chart_grp = "results/waterbirds_group_accuracies.png"
    if os.path.exists(chart_grp):
        s11.shapes.add_picture(chart_grp, Inches(6.6), Inches(1.8), width=Inches(5.9))

    set_speaker_notes(s11, "Apresentador 3", "1.5 min",
        "Neste slide da Seção V, mostramos o gráfico de acurácia por grupo de teste. "
        "Observem as barras do ERM: no Grupo 0 (barra azul escura), ele acerta quase 100%. "
        "Mas no Grupo 2 (ave aquática em terra), a acurácia cai para 67%. "
        "Agora olhem as barras verdes do DFR: elas formam uma linha quase perfeitamente horizontal em 90% em todos os 4 grupos! "
        "Houve uma leve redução no Grupo 0 de 99% para 92%, mas isso é exatamente o que esperamos ao eliminar a memorização do atalho. "
        "O ganho de 22 pontos percentuais no Grupo 2 compensou amplamente essa variação, elevando inclusive a acurácia média geral de 87% para 90%.")

    # =========================================================================
    # SLIDE 12: SEÇÃO VI - PROVANDO O ATALHO NO BASELINE ERM (PASSO 1)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Provando o Atalho: Onde o Modelo Baseline Olha Antes dos Ajustes?", "Seção VI: Interpretabilidade Visual", "Apresentador 3")

    img_baseline_proof = "results/waterbirds_gradcam_baseline_bias.png"
    if os.path.exists(img_baseline_proof):
        s12.shapes.add_picture(img_baseline_proof, Inches(0.8), Inches(1.75), width=Inches(11.733))

    cap_box_12 = s12.shapes.add_textbox(Inches(0.8), Inches(6.1), Inches(11.733), Inches(0.9))
    tf_c12 = cap_box_12.text_frame
    tf_c12.word_wrap = True
    p_c12 = tf_c12.paragraphs[0]
    p_c12.text = (
        "• PROVA INQUESTIONÁVEL DO VIÉS NO MODELO ORIGINAL (SEM AJUSTES):\n"
        "  - Amostra Real de Teste: Gaivota (Waterbird) pousada em floresta de bambu (Terra) — Grupo Minoritário G2.\n"
        "  - Comportamento do ERM: O mapa de ativação (Grad-CAM) ignora completamente a ave e concentra todo o calor nos caules de bambu.\n"
        "  - Diagnóstico: Como o fundo é terra, o ERM classifica com convicção como 'Landbird' (94.9% de certeza!). Isso prova fisicamente o atalho espúrio antes de qualquer intervenção!"
    )
    p_c12.font.size = Pt(11.5)
    p_c12.font.color.rgb = COLOR_TEXT_DARK

    set_speaker_notes(s12, "Apresentador 3", "2.0 min",
        "Atenção a este slide da Seção VI: antes de apresentar qualquer solução, precisamos PROVAR que o modelo original realmente sofre do viés de atalho. "
        "Aqui avaliamos o modelo Baseline ERM normal, exatamente como saiu do treinamento tradicional, sem nenhum filtro ou rebalanceamento. "
        "Pegamos uma imagem de teste real do Grupo 2: uma gaivota (uma ave 100% aquática) inserida em uma floresta de bambu. "
        "Vejam o painel da direita: o mapa de calor Grad-CAM mostra em vermelho exatamente onde a rede colocou sua atenção para tomar a decisão. "
        "O calor não está na cabeça nem no corpo da gaivota: está inteiramente espalhado pelos caules de bambu acima e abaixo da ave! "
        "A rede olhou para os caules de bambu, achou que vegetação significa ave terrestre e cravou a resposta errada: 'Landbird' com quase 95% de certeza! "
        "Essa é a prova visual cabal e inquestionável de que o modelo baseline foi enganado pelo fundo espúrio.")

    # =========================================================================
    # SLIDE 13: SEÇÃO VI - REVERTENDO O ATALHO COM DFR (PASSO 2: ANTES VS DEPOIS)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Revertendo o Atalho: A Mudança de Atenção do ERM para o DFR", "Seção VI: Interpretabilidade Visual", "Apresentador 3")

    cam_img = "results/waterbirds_gradcam_comparison.png"
    if os.path.exists(cam_img):
        s13.shapes.add_picture(cam_img, Inches(0.8), Inches(1.75), width=Inches(11.733))

    cap_box_13 = s13.shapes.add_textbox(Inches(0.8), Inches(6.1), Inches(11.733), Inches(0.9))
    tf_c13 = cap_box_13.text_frame
    tf_c13.word_wrap = True
    p_c13 = tf_c13.paragraphs[0]
    p_c13.text = (
        "• A TRANSFORMAÇÃO RADICAL (ANTES vs. DEPOIS):\n"
        "  - ERM (Centro): Foco espalhado na vegetação ao fundo -> Predição Errada: Landbird (94.9%).\n"
        "  - DFR (Direita): Ao reponderar a camada linear em dados balanceados, a rede 'desliga' os filtros de bambu e concentra o calor estritamente na cabeça, bico e corpo da gaivota -> Predição Correta: Waterbird (97.9%)!"
    )
    p_c13.font.size = Pt(11.5)
    p_c13.font.color.rgb = COLOR_TEXT_DARK

    set_speaker_notes(s13, "Apresentador 3", "2.0 min",
        "Agora que provamos o viés no modelo original, vejam o que acontece após aplicarmos o DFR. "
        "Este slide mostra a comparação direta: Imagem Original, Modelo ERM antes do ajuste, e Modelo DFR após o ajuste. "
        "Lembrem-se: nós NÃO retreinamos a ResNet-50. A espinha dorsal convolucional é rigorosamente a mesma! "
        "O que mudou? O classificador linear final agora recebeu pesos otimizados em um conjunto de validação balanceado. "
        "Vejam no painel da direita como a atenção migrou: os caules de bambu foram totalmente apagados do mapa de calor! "
        "O calor vermelho agora está perfeitamente delimitado na cabeça, no bico e no peito da gaivota. "
        "A predição foi corrigida com sucesso para 'Waterbird' com 97.9% de certeza! "
        "Isso confirma experimentalmente a hipótese do DFR: a rede já possuía a capacidade de ver a ave, bastava silenciar a interferência da camada linear.")

    # =========================================================================
    # SLIDE 14: SEÇÃO VII - DISCUSSÃO E LIMITAÇÕES
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Discussão Crítica: Limitações do DFR e Desafios no Mundo Real", "Seção VII: Discussão & Conclusão", "Apresentador 3")

    disc_cards = [
        ("Dependência de Validação", "Requisito de Rótulos de Grupo",
         "• O DFR requer um pequeno conjunto de validação com anotações de grupo (classe + atributo espúrio).\n"
         "• Se o viés for desconhecido a priori (ex: etnia, modelo de câmera ou artefato não catalogado), balancear a validação torna-se mais difícil.",
         COLOR_ACCENT_RED),
        ("Atalhos Multidimensionais", "Além do Fundo Binário",
         "• No Waterbirds, o atalho é binário e explícito (terra vs. água).\n"
         "• Em aplicações reais médicas ou industriais, múltiplos fatores de confusão atuam simultaneamente (ex: ângulo da foto, iluminação e ruído do sensor).",
         COLOR_PRIMARY),
        ("Novas Fronteiras de Pesquisa", "Descoberta Não Supervisionada",
         "• Pesquisas recentes combinam DFR com clustering no espaço latente (ex: George, Sohoni et al., 2020) para descobrir grupos espúrios automaticamente.\n"
         "• Isso viabiliza o DFR mesmo quando não temos anotações prévias de atributos de grupo.",
         COLOR_ACCENT_GREEN)
    ]

    for i, (title, sub, body, col) in enumerate(disc_cards):
        shape = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i * 4.04), Inches(1.8), card_w3, card_h3)
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD_BG
        shape.line.color.rgb = COLOR_CARD_BORDER
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(4)

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(11)
        p_s.font.bold = True
        p_s.font.color.rgb = COLOR_TEXT_MUTED
        p_s.space_after = Pt(12)

        p_t = tf.add_paragraph()
        p_t.text = body
        p_t.font.size = Pt(11.5)
        p_t.font.color.rgb = COLOR_TEXT_DARK

    set_speaker_notes(s14, "Apresentador 3", "1.5 min",
        "Na Seção VII de Discussão e Limitações, fazemos uma reflexão crítica sobre as fronteiras do método. "
        "A principal limitação do DFR é assumir que temos um conjunto de validação pequeno com rótulos de grupo (no nosso caso, 133 amostras por grupo). "
        "Se não soubermos com antecedência qual é o atalho que a rede está explorando, não conseguimos balancear a validação manualmente. "
        "Além disso, no mundo real os atalhos raramente são puramente binários como terra versus água: muitas vezes são contínuos e multivariados. "
        "Felizmente, métodos recentes combinam DFR com algoritmos de agrupamento não supervisionado no espaço latente para contornar essa exigência.")

    # =========================================================================
    # SLIDE 15: SEÇÃO VII - CONCLUSÃO & LIÇÕES APRENDIDAS
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Conclusão: Três Lições Fundamentais do Projeto", "Seção VII: Discussão & Conclusão", "Apresentadores 1, 2 e 3")

    concl_items = [
        ("1. A Acurácia Média Mascara Falhas Críticas",
         "O ERM alcançou 87.88% de média geral com 99.56% no grupo comum, mas desabou para 67.45% no grupo minoritário. Avaliar modelos críticos com a Worst-Group Accuracy (WGA) é indispensável para evitar falhas em produção.",
         COLOR_ACCENT_RED),
        ("2. Redes Profundas Aprendem Representações Robustas",
         "O shortcut learning não significa que a rede convolucional foi corrompida. As representações latentes profundas preservam as características essenciais da ave; o erro reside na combinação linear da camada final.",
         COLOR_PRIMARY),
        ("3. Simplicidade e Eficiência Computacional (DFR)",
         "Retreinar apenas a camada linear final com dados de validação balanceados elevou o WGA de 67.45% para 89.56% em menos de 1 segundo, provando que soluções simples e convexas superam retreinamentos caros e instáveis.",
         COLOR_ACCENT_GREEN)
    ]

    for i, (title, desc, col) in enumerate(concl_items):
        shape = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8 + i * 1.6), Inches(11.733), Inches(1.35))
        shape.fill.solid()
        shape.fill.fore_color.rgb = COLOR_CARD_BG
        shape.line.color.rgb = COLOR_CARD_BORDER
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.2)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(4)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = COLOR_TEXT_DARK

    set_speaker_notes(s15, "Apresentadores 1, 2 e 3", "1.0 min",
        "[Apresentador 1]: Para encerrar nosso seminário, deixamos três conclusões centrais: "
        "Primeiro, nunca confiem em acurácia média quando houver assimetria de classes e atributos nos dados. "
        "[Apresentador 2]: Segundo, o problema de shortcut learning é em grande parte um problema de ponderação final e não de incapacidade representacional do extrator convolucional. "
        "[Apresentador 3]: E terceiro, o DFR comprova que com elegância matemática e validação balanceada, podemos alcançar modelos robustos em frações de segundo. "
        "Agradecemos ao professor e aos colegas e encerramos com nossas referências bibliográficas.")

    # =========================================================================
    # SLIDE 16: SEÇÃO VIII - REFERÊNCIAS BIBLIOGRÁFICAS
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Referências Bibliográficas Auditadas", "Seção VIII: Referências", "Todos os Apresentadores")

    tb_refs = s16.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.0))
    tf_r = tb_refs.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0

    refs_list = [
        ("Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020).", 
         "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization.", 
         "International Conference on Learning Representations (ICLR 2020). [Criação do benchmark Waterbirds e proposta do Group DRO]"),
        ("Kirichenko, P., Izmailov, P., & Wilson, A. G. (2022).", 
         "Last Layer Re-Training is Sufficient for Robustness.", 
         "Advances in Neural Information Processing Systems (NeurIPS 2022). [Proposta do método Deep Feature Reweighting - DFR]"),
        ("Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020).", 
         "Shortcut learning in deep neural networks.", 
         "Nature Machine Intelligence, 2(11), 665-673. [Artigo seminal sobre o conceito e taxonomia de shortcut learning]"),
        ("Pezeshki, M., Kaba, O., Bengio, Y., Courville, A. C., Precup, D., & Lajoie, G. (2021).", 
         "Gradient Starvation: A Mechanism for Representation Learning in Neural Networks.", 
         "Advances in Neural Information Processing Systems (NeurIPS 2021). [Mecânica de estrangulamento de gradientes]"),
        ("Liu, E. Z., Haghgoo, B., Chen, A. S., Raghunathan, A., Koh, P. W., Sagawa, S., Liang, P., & Finn, C. (2021).", 
         "Just Train Twice: Improving Group Robustness without Attribute Annotations.", 
         "International Conference on Machine Learning (ICML 2021). [Proposta do método JTT]"),
        ("Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).", 
         "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.", 
         "IEEE International Conference on Computer Vision (ICCV 2017), pp. 618-626. [Técnica de explicabilidade e mapas de atenção visual]"),
        ("Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011).", 
         "The Caltech-UCSD Birds-200-2011 Dataset.", 
         "California Institute of Technology. [Imagens das aves utilizadas no Waterbirds]"),
        ("Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2017).", 
         "Places: A 10 Million Image Database for Scene Recognition.", 
         "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). [Imagens de cenários terrestres e aquáticos]")
    ]

    for i, (authors, title, venue) in enumerate(refs_list):
        p = tf_r.paragraphs[0] if i == 0 else tf_r.add_paragraph()
        p.text = f"[{i+1}] {authors} \"{title}\" {venue}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(5)

    set_speaker_notes(s16, "Todos", "-",
        "Slide de referências bibliográficas auditadas. Todos os artigos listados foram diretamente citados e utilizados "
        "no embasamento teórico, na formulação das hipóteses e na execução do código do nosso trabalho.")

    out_pptx = "seminario_vies_redes_neurais.pptx"
    prs.save(out_pptx)
    print(f"Apresentação com prova do Baseline ERM e comparação com DFR gerada em: {out_pptx}")

if __name__ == "__main__":
    create_deck()
