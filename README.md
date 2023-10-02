
![CAPA - Gif - Detecção de Deep Fakes](https://github.com/NathFarinha/TCC_DeepFake_Detection_v1/assets/89995351/1d8e7ed7-ed14-4cec-ba0c-3392e3cead90)

# ANÁLISE DE MODELOS DETECTORES DE DEEPFAKE USANDO APRENDIZADO PROFUNDO

# Índice 

* [Visão Geral](#visao-geral)
* [Funcionalidades](#funcionalidades)
* [Acesso ao Projeto](#acesso-ao-projeto)

## Visão Geral

Este é um projeto de detecção de deepfakes que visa identificar vídeos e imagens falsificadas usando técnicas de aprendizado profundo. O objetivo deste projeto é aumentar a conscientização sobre a ameaça dos deepfakes e fornecer uma ferramenta para sua detecção.
Foram analisado algumas redes EfficientNet e a Xception, a fim de comparar seus resultados na detecção de DeepFakes.

Inspirado no projeto: https://github.com/polimi-ispl/icpr2020dfdc.git

![Detecção de Deep Fakes - APP Streamlit - GIF demonstracao](https://github.com/NathFarinha/TCC_DeepFake_Detection_v1/assets/89995351/d35e95eb-dad7-442f-b9c9-3e70b8b2dc93)

## Funcionalidades

- Detecção de deepfakes em vídeos e imagens
- Suporte para vários modelos de aprendizado de máquina (Xception, EfficientNetB4, EfficientNetB4ST, EfficientNetAutoAttB4, EfficientNetAutoAttB4ST)
- Integração com diferentes fontes de dados para treinamento e teste
- Notebooks para geração e análise dos resultados

## Acesso ao Projeto
Códigos e notebooks relevantes para executar o APP Streamlit para detecção de deepfakes e realizar os testes nos modelos pré-treinados.

#### APP Streamlit
<a target="_blank" href="https://colab.research.google.com/github/NathFarinha/TCC_DeepFake_Detection_v1/blob/56409b5ed1648be9fa21e5a6483e79ba8dab3388/Colab%20Notebooks/PREDICTIONS/Streamlit_APP.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Gerar resultados dos modelos pré-treinados nos 1000 vídeos de teste
<a target="_blank" href="https://colab.research.google.com/github/NathFarinha/TCC_DeepFake_Detection_v1/blob/c9418e0833c70102c7e056846653ef70885a2566/Colab%20Notebooks/RESULTADOS/Generate_results_DFDC.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Gerar resultados das redes individuais - Matriz confusão / Curva ROC
<a target="_blank" href="https://colab.research.google.com/github/NathFarinha/TCC_DeepFake_Detection_v1/blob/2e9ec9cb20d6e3caeb7105abb80ce61ea01c644f/Colab%20Notebooks/RESULTADOS/Analyze_results.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Gerar resultados das fusões das redes
<a target="_blank" href="https://colab.research.google.com/github/NathFarinha/TCC_DeepFake_Detection_v1/blob/2e9ec9cb20d6e3caeb7105abb80ce61ea01c644f/Colab%20Notebooks/RESULTADOS/Analyze_results_net_fusion_paper.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>





