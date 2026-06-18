# Detecção Inteligente de Dispositivos Móveis e Classificação de Ameaças em Tempo Real via Algoritmo YOLOv8 e Filtragem Histerética Temporal

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![W&B Rastreamento](https://img.shields.io/badge/Weights%20%26%20Biases-Ativo-gold.svg)](https://wandb.ai/)
[![ONNX](https://img.shields.io/badge/ONNX-Exportable-green.svg)](https://onnx.ai/)

Este repositório apresenta o desenvolvimento de um pipeline completo de Visão Computacional voltado para a detecção de telefones celulares e a classificação automática de situações de risco (*THREAT* ou Ameaça) em fluxos de vídeo. O projeto evolui as abordagens convencionais de inferência frame a frame ao introduzir filtragem temporal baseada em histerese para mitigação de oscilações (*flickering*), avaliação robusta por meio de varredura (*sweep*) de limiar e estratégias de mineração de exemplos difíceis (*hard negatives*), preparando o modelo para aplicações de nível de produção e validação em artigos científicos.

---

## 1. Introdução e Contextualização Científica

Em ambientes controlados de alta segurança (tais como salas de avaliação acadêmica, laboratórios de pesquisa corporativos, postos de operação industrial ou condução veicular), a utilização não autorizada de dispositivos móveis constitui uma violação de conformidade e um vetor de vazamento de dados ou acidentes. 

Os métodos tradicionais baseados em classificadores estáticos apresentam duas limitações críticas:
1. **Falta de Contexto Espacial**: A mera presença de um smartphone (por exemplo, pousado sobre uma mesa) não caracteriza necessariamente um uso ativo ou uma infração.
2. **Instabilidade Temporal**: Variações na iluminação, oclusões parciais ou desfoques de movimento geram detecções intermitentes (falsos positivos e falsos negativos consecutivos de curtíssima duração), o que inviabiliza o acionamento de alarmes ou o bloqueio físico de telas devido ao alto nível de ruído visual.

Este projeto resolve ambas as limitações propondo uma arquitetura de duas camadas:
* **Camada de Detecção e Heurística Geométrica**: Um detector baseado em aprendizado profundo (YOLOv8) localiza a classe `mobile_phone`. Em seguida, regras baseadas nas proporções espaciais da caixa delimitadora (*bounding box*) determinam se o dispositivo está em posição de uso ativo (Ameaça).
* **Camada de Filtro Temporal Histerético**: Uma janela deslizante analisa o histórico recente de predições, exigindo consistência estatística antes de alterar o estado de alerta do sistema (bloqueio da tela por borramento).

---

## 2. Estrutura do Repositório

O repositório está organizado nos seguintes notebooks e arquivos fundamentais:

*   [AP2_Visão_Computacional_Detecção_de_Telefones.ipynb](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/AP2_Visão_Computacional_Detecção_de_Telefones.ipynb): **Notebook principal (Versão Atual - AP2)** contendo o pipeline expandido: integração com Weights & Biases (W&B), otimização de threshold via F1-Score, filtragem temporal com histerese, coleta ativa de *hard negatives* e exportação com benchmark ONNX.
*   [AP1_Visão_Computacional_Detecção_de_Telefones.ipynb](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/AP1_Visão_Computacional_Detecção_de_Telefones.ipynb): Notebook de desenvolvimento da primeira fase (AP1), contendo a lógica preliminar e o protótipo inicial do modelo.
*   [Collab_AP1_COMPUTER_VISION_Detecção_de_Telefones_Breno.ipynb](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/Collab_AP1_COMPUTER_VISION_Detecção_de_Telefones_Breno.ipynb): Versão alternativa e colaborativa focada nos experimentos iniciais do grupo.
*   `README.md`: Documentação científica e guia de reprodutibilidade do projeto.

---

## 3. Arquitetura do Pipeline

A figura abaixo descreve conceitualmente o fluxo de dados desde a preparação dos dados brutos até a execução em tempo real e benchmark:

```mermaid
flowchart TD
    A[Dataset Pascal VOC XML] --> B[Conversor XML para YOLO TXT]
    B --> C[Divisão Estatística Treino/Val 80/20]
    C --> D[Treinamento YOLOv8n imgsz=800]
    D -->|Rastreamento de Métricas| E[Weights & Biases W&B]
    D --> F[Modelo Treinado best.pt]
    F --> G[Regra de Negócio Geométrica is_threat]
    F --> H[Sweep de Threshold conf F1-Score]
    F --> I[Mineração de Hard Negatives]
    F --> J[Exportação ONNX e Benchmark de Latência]
    G --> K[Filtro Temporal Histerético]
    K --> L[Inferencia Real-time Webcam/Vídeo com Blur e Logs CSV]
```

---

## 4. Metodologia e Preparação de Dados

### 4.1. Conversão e Normalização de Anotações
O dataset original encontra-se rotulado no formato Pascal VOC, em que as coordenadas de delimitação de objetos são absolutas em relação aos pixels da imagem: $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$. 

Para a alimentação eficiente da rede YOLOv8, estas coordenadas são normalizadas para valores reais no intervalo $[0.0, 1.0]$, em termos de centro do retângulo $(x_c, y_c)$, largura ($w$) e altura ($h$), por meio das equações:

$$x_c = \frac{x_{\min} + x_{\max}}{2 \cdot W_{\text{imagem}}}$$

$$y_c = \frac{y_{\min} + y_{\max}}{2 \cdot H_{\text{imagem}}}$$

$$w = \frac{x_{\max} - x_{\min}}{W_{\text{imagem}}}$$

$$h = \frac{y_{\max} - y_{\min}}{H_{\text{imagem}}}$$

A função Python [convert_voc_to_yolo](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/AP2_Visão_Computacional_Detecção_de_Telefones.ipynb#L4) implementa este cálculo mitigando divisões por zero ou estouros de limites.

### 4.2. Divisão e Particionamento de Dados
A estruturação do dataset segue a divisão padrão para aprendizado de máquina supervisionado. Com uma semente de pseudorrandomização fixa (`random.seed(42)`), as imagens e suas respectivas anotações convertidas em arquivos `.txt` são distribuídas na seguinte proporção:
*   **Treinamento (80%)**: Utilizado para o ajuste de pesos dos parâmetros do modelo.
*   **Validação (20%)**: Utilizado para o monitoramento de sobreajuste (*overfitting*) e validação de métricas de generalização.

---

## 5. Configurações e Treinamento do Modelo

### 5.1. Seleção da Rede Neural
Adota-se a arquitetura **YOLOv8n** (YOLOv8 Nano), que possui aproximadamente 3,2 milhões de parâmetros. Esta escolha se justifica pelo foco em dispositivos de borda ou execução em CPU em tempo real, onde uma baixa latência computacional é prioritária em relação a backbones massivos.

### 5.2. Hiperparâmetros de Treinamento (Fase AP2)
Comparado à fase AP1 (que empregava resolução padrão de 640 pixels), a fase AP2 adota hiperparâmetros otimizados para melhoria do mAP em objetos de escala reduzida:

*   **Tamanho de imagem (`imgsz`)**: $800 \times 800$ pixels. Aumentar a escala de entrada de 640 para 800 melhora a precisão na detecção de celulares posicionados a maiores distâncias em relação à câmera.
*   **Épocas de Treinamento (`epochs`)**: $30$.
*   **Tamanho do lote (`batch`)**: $16$ amostras por iteração.
*   **Modelo Base**: `yolov8n.pt` (transfer learning a partir do dataset MS COCO).

### 5.3. Integração com Weights & Biases (W&B)
Durante a fase de otimização, o pipeline estabelece comunicação automática com o painel do **Weights & Biases**. Isso permite o monitoramento síncrono das funções de perda (*box loss*, *class loss* e *dfl loss*) e das curvas de precisão-revocação na nuvem, garantindo auditoria completa do comportamento do modelo durante as 30 épocas.

---

## 6. Heurística Geométrica de Classificação de Ameaça (Regra de Negócio)

Após o YOLOv8 retornar as caixas delimitadoras de celulares detectados, cada caixa passa por uma verificação de regras geométricas descritas na função [is_threat](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/AP2_Visão_Computacional_Detecção_de_Telefones.ipynb#L14). O objetivo é isolar celulares que estão sendo ativamente empunhados e visualizados pelo usuário.

Seja a caixa delimitadora definida pelas coordenadas absolutas $(x_1, y_1)$ e $(x_2, y_2)$ em uma imagem de resolução $W \times H$:

1.  **Centroide da Bounding Box**:
    $$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$$
2.  **Área da Bounding Box**:
    $$A_{\text{box}} = (x_2 - x_1) \cdot (y_2 - y_1)$$
3.  **Proporção de Área Relativa**:
    $$\text{Ratio}_{\text{area}} = \frac{A_{\text{box}}}{W \cdot H}$$
4.  **Razão de Aspecto (Aspect Ratio)**:
    $$\text{Ratio}_{\text{aspect}} = \frac{y_2 - y_1}{\max(x_2 - x_1, 1)}$$

A classificação retorna `True` (Ameaça) se e somente se as seguintes **quatro restrições** forem atendidas de forma simultânea:

$$\begin{cases}
0.30 \cdot W \le c_x \le 0.70 \cdot W & \text{(Centração Horizontal: Celular no centro da cena)} \\
\text{Ratio}_{\text{area}} \ge 0.03 & \text{(Área Mínima: Próximo o suficiente para indicar uso)} \\
c_y \le 0.85 \cdot H & \text{(Limite Vertical: Celular levantado, não apoiado sobre a mesa)} \\
\text{Ratio}_{\text{aspect}} \ge 1.2 & \text{(Orientação em Retrato: Smartphone posicionado verticalmente)}
\end{cases}$$

Se qualquer condição falhar, o objeto detectado permanece como uma detecção comum de celular segura (`mobile_phone`).

---

## 7. Sistema de Filtragem Histerética Temporal

Para eliminar o falso acionamento decorrente do *flickering* do detector, implementou-se o algoritmo `TemporalThreatFilter` operando em janela deslizante de decisões de tamanho fixo $N=20$.

### 7.1. Modelagem Matemática do Filtro
Seja $x_t \in \{0, 1\}$ a saída da heurística geométrica de ameaça no frame $t$:

$$x_t = \begin{cases} 
1 & \text{se } \text{is\_threat} \text{ retorna True} \\ 
0 & \text{caso contrário} 
\end{cases}$$

O histórico de decisões recentes é mantido em uma estrutura de fila de dupla extremidade com capacidade máxima limitada a $N$:

$$W_t = \{x_{t-N+1}, x_{t-N+2}, \dots, x_t\}$$

O **Score Temporal** ($S_t$) no instante $t$ é computado como a média aritmética simples da janela:

$$S_t = \frac{1}{|W_t|} \sum_{i \in W_t} i$$

O estado final do alerta de travamento ($A_t \in \{0, 1\}$) é governado por uma lógica de transição com **histerese**, dependendo do estado anterior $A_{t-1}$:

$$A_t = \begin{cases}
1 & \text{se } S_t \ge \tau_{\text{trigger}} \\
0 & \text{se } S_t < \tau_{\text{release}} \\
A_{t-1} & \text{se } \tau_{\text{release}} \le S_t < \tau_{\text{trigger}}
\end{cases}$$

Adotam-se os seguintes limiares heurísticos baseados em testes empíricos:
*   $\tau_{\text{trigger}} = 0.60$: O alerta é ativado apenas se pelo menos 60% dos frames na janela contiverem ameaças geométricas ativas.
*   $\tau_{\text{release}} = 0.30$: O alarme é desligado (liberação da tela) apenas quando a densidade de ameaças cai abaixo de 30%.

### 7.2. Ação de Segurança e Registro de Evidências
Quando $A_t = 1$:
1.  **Obscurecimento Visual**: Aplica-se um filtro gaussiano de desfoque com kernel $99 \times 99$ e desvio padrão $\sigma = 30$ em todo o frame, seguido pela mistura com um canal preto translúcido para diminuir a visibilidade:
    $$\text{Frame}_{\text{saída}} = 0.5 \cdot \text{Blur}(\text{Frame}, 99) + 0.5 \cdot \text{Overlay}_{\text{preto}}$$
2.  **Mensagens Visuais**: São inseridos avisos de "TELA BLOQUEADA" e "CELULAR DETECTADO" no centro da tela.
3.  **Auditoria Automatizada**: Caso o temporizador de *cooldown* (padrão: 45 frames) esteja zerado, o frame de evidência é persistido em disco e um log de evento é anexado a um arquivo CSV (`threat_events.csv`), contendo:
    *   Índice do Frame
    *   Estampa Temporal (segundos)
    *   Score de Alerta Temporal ($S_t$)
    *   Confiança Máxima detectada pelo YOLOv8
    *   Caminho de destino do frame de evidência.

---

## 8. Avaliação Robusta e Otimização de Limiar (Sweep)

O projeto realiza duas etapas de validação para garantir a precisão científica dos resultados de detecção.

### 8.1. Métricas Globais Oficiais
Utilizando a API `model.val()` do Ultralytics no conjunto de dados de validação, extraem-se as seguintes métricas a um limiar padrão (IoU de intersecção a 0.6):
*   **mAP@50**: Precisão Média Média a IoU fixo de 0.5.
*   **mAP@50-95**: Média aritmética de mAP variando o IoU de 0.5 a 0.95 com passo de 0.05.
*   **Precision (Precisão)**: Proporção de detecções positivas corretas.
*   **Recall (Revocação/Sensibilidade)**: Proporção de objetos reais detectados com sucesso.

### 8.2. Otimização do Limiar de Confiança via F1-Score (Sweep de Threshold)
Para encontrar o equilíbrio ótimo entre falsos positivos e falsos negativos, a função [evaluate_thresholds](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/AP2_Visão_Computacional_Detecção_de_Telefones.ipynb#L25) faz um varrimento de thresholds de confiança variando de $0.10$ a $0.95$ com intervalos discretos de $0.05$.

Para cada valor de limiar de confiança, as caixas preditas são cruzadas com as anotações reais do conjunto de validação. O emparelhamento ocorre calculando a Intersecção sobre União (IoU):

$$\text{IoU} = \frac{\text{Área}(\text{Box}_{\text{pred}} \cap \text{Box}_{\text{real}})}{\text{Área}(\text{Box}_{\text{pred}} \cup \text{Box}_{\text{real}})}$$

Uma predição é rotulada como **Verdadeiro Positivo (TP)** se seu $\text{IoU} \ge 0.5$ com um rótulo real não utilizado. Predições sobressalentes geram **Falsos Positivos (FP)**, e rótulos reais não emparelhados tornam-se **Falsos Negativos (FN)**.

Com base nestes valores, o notebook computa o perfil de $F_1\text{-Score}$ e seleciona automaticamente o limiar de confiança ótimo que maximiza o $F_1$:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

A curva resultante (Confidence Threshold vs Score) é exibida em gráfico gerado pela biblioteca Matplotlib.

---

## 9. Mineração Ativa de Hard Negatives

Com vistas ao aprendizado contínuo do modelo, desenvolveu-se a função [collect_hard_negatives](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/AP2_Visão_Computacional_Detecção_de_Telefones.ipynb#L29).

Ela analisa imagens do conjunto de validação que **não contêm anotações originais** de celular (imagens vazias para a classe avaliada). Se o modelo YOLOv8 realizar uma detecção com confiança superior a um limiar específico (ex: 0.35) nessa imagem, isso é categorizado como um **Falso Positivo de alta confiança** (exemplo difícil ou *Hard Negative*). 

Essas imagens de erro são isoladas e salvas em uma pasta `/content/hard_negatives`. Em ciclos de refinamento do artigo, elas devem ser integradas ao conjunto de treino sem anotações (arquivos `.txt` vazios correspondentes), instruindo explicitamente a rede a mitigar predições errôneas sob contextos de fundo semelhantes.

---

## 10. Exportação ONNX e Benchmark de Latência

Buscando independência do framework PyTorch em ambientes de produção, o modelo treinado é exportado para o formato aberto **ONNX** (Open Neural Network Exchange), utilizando simplificação de grafo (`simplify=True`) e suporte a dimensões de lote dinâmicas (`dynamic=True`):

```python
onnx_path = best_model.export(format="onnx", dynamic=True, simplify=True)
```

O pipeline avalia de forma quantitativa a performance comparativa executando um teste de estresse nas primeiras 100 imagens de validação. Computam-se:
*   **Latência Média por Frame ($L_{\text{média}}$)** em milissegundos.
*   **Latência no Percentil 95 ($L_{\text{p95}}$)**, métrica crítica para garantir a ausência de gargalos de processamento esporádicos.
*   **Frames Por Segundo Estimados (FPS)**:
    $$\text{FPS} = \frac{1000}{L_{\text{média}}}$$

Se o hardware local disponibilizar drivers adequados, o notebook realiza a comparação automatizada via terminal através do comando:

```bash
yolo benchmark model=runs/mobile_phone_detector/weights/best.pt imgsz=800 half=False data=mobile_phone_yolo/data.yaml device=cpu
```

Isso realiza um benchmark completo medindo consumo de memória e velocidade entre os backbones PyTorch, ONNX e TensorRT.

---

## 11. Instruções para Reprodutibilidade e Execução

### 11.1. Execução via Google Colab (Recomendado)
1.  Faça o upload do notebook [AP2_Visão_Computacional_Detecção_de_Telefones.ipynb](file:///c:/Users/User/Documents/GitHub/Deteccao-Celulares-YOLOv8/AP2_Visão_Computacional_Detecção_de_Telefones.ipynb) para o seu ambiente do Google Drive.
2.  Garanta que os arquivos compactados do seu conjunto de dados (`100_imagens_celulares.zip`, `520_imagens_celulares.zip` ou `1300_imagens_celulares.zip`) estejam carregados no diretório `/content`.
3.  Execute as células sequencialmente. Ao atingir o treinamento, siga o prompt de autenticação do Weights & Biases para salvar as estatísticas de validação em sua conta.
4.  Após a conclusão, os arquivos de resultados e logs estarão disponíveis em `/content/threat_evidence` e `/content/threat_events.csv`.

### 11.2. Execução em Ambiente Local (Webcam ativa)
Para rodar o pipeline de detecção em tempo real consumindo a webcam física do seu computador:

1.  Clone o repositório localmente.
2.  Instale os pré-requisitos necessários via terminal:
    ```bash
    pip install ultralytics opencv-python numpy pandas matplotlib seaborn scikit-learn onnx onnxruntime
    ```
3.  Abra o notebook em seu ambiente local (Jupyter Notebook ou VS Code) e altere a chamada de predição do Colab para a sua câmera local nas últimas células executáveis:
    ```python
    from ultralytics import YOLO
    
    # Carregue o modelo treinado local
    model = YOLO("caminho/para/seu/best.pt")
    
    # Executa detecção com webcam local (source=0)
    run_realtime_detection(model, source=0, conf_thr=0.35)
    ```

---

## 12. Saídas e Resultados do Pipeline

Durante e após a execução do pipeline de processamento da AP2, são criados os seguintes artefatos:

*   **`best.pt` e `best.onnx`**: Pesos binários do modelo treinado otimizado nos formatos nativos PyTorch e ONNX, respectivamente.
*   **`threat_detection_results/`**: Amostras de validação estáticas rotuladas com as marcações de segurança `SAFE` ou `THREAT` e desfoque gaussiano de tela.
*   **`threat_realtime_output.mp4`**: Gravação em vídeo do fluxo processado apresentando a sobreposição do alerta temporal histerético e a ocultação da tela sob alta probabilidade de infração.
*   **`threat_evidence/`**: Capturas de tela dos instantes de infração detectados (organizados de forma sequencial com índice de frame).
*   **`threat_events.csv`**: Base de dados em planilha de eventos, contendo o log estruturado para auditorias científicas do detector de celulares.
*   **`hard_negatives/`**: Conjunto de imagens salvas para o ciclo subsequente de retreinamento do detector.

---

## 13. Conclusão e Autoria

Este ecossistema foi desenvolvido como parte integrante das atividades práticas de **Visão Computacional** (**AP2**), servindo como fundamentação experimental e metodológica para a redação de artigos científicos na subárea de segurança inteligente baseada em aprendizagem profunda.

*Desenvolvido para fins de validação acadêmica e pesquisa em Visão Computacional aplicada.*
