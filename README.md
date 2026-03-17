# GSOM para Classificação de Logs de Firewall em Fluxos Contínuos de Dados

Este repositório reúne o código e os dados utilizados na pesquisa de mestrado sobre o uso de **Growing Self-Organizing Maps (GSOM)** para classificação de logs de firewall em **Fluxos Contínuos de Dados (FCDs)**.

O objetivo do trabalho foi investigar a capacidade do GSOM de atuar em um cenário com fluxo de dados, adaptação temporal e ausência de rótulos imediatos durante a fase online, comparando seu comportamento com abordagens supervisionadas de referência.

## Contexto

Logs de firewall refletem decisões operacionais como permitir, negar, descartar ou interromper conexões de rede. Em ambientes reais, esses dados chegam continuamente e podem sofrer alterações ao longo do tempo, seja por mudanças de política, reconfiguração de regras ou surgimento de novos padrões de tráfego.

Neste contexto, o trabalho avalia o uso do GSOM como alternativa para representar e classificar padrões em fluxo contínuo, explorando sua capacidade de auto-organização topológica e adaptação estrutural.

## Estrutura do repositório

```text
gsom-firewall/
├── Coletas/
├── GSOM.py
└── README.md
