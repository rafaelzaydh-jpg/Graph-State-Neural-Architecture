# Graph-State Neural Architecture

## Uma arquitetura de modelo de linguagem baseada em grafos, computação esparsa, memória hierárquica e parâmetros pagináveis

**White Paper — Versão 0.1**  
**Status:** Proposta arquitetural e plano experimental

---

## Resumo

Os modelos de linguagem modernos alcançaram capacidades extraordinárias principalmente por meio do aumento do número de parâmetros, dados e capacidade computacional. Entretanto, a arquitetura predominante continua baseada em uma premissa fundamental: uma sequência atravessa uma estrutura neural essencialmente fixa, enquanto contexto, parâmetros e estados intermediários são mantidos próximos ao mecanismo de computação.

Essa abordagem produz três formas distintas de pressão sobre o sistema:

1. o custo computacional associado ao processamento de sequências longas;
2. o crescimento da memória necessária para manter contexto e estado de inferência;
3. a necessidade de manter uma parcela muito grande dos parâmetros do modelo em memória de alta velocidade.

Este trabalho propõe uma arquitetura alternativa denominada provisoriamente **Graph-State Neural Architecture (GSNA)**.

Na GSNA, um modelo de linguagem não é representado primariamente como uma sequência fixa de camadas. Ele é representado como um **grafo neural esparso e dinamicamente roteado**, composto por unidades computacionais especializadas, estados neurais, memória persistente e mecanismos explícitos de roteamento.

A cada etapa, somente uma pequena região desse grafo precisa estar ativa.

O modelo completo pode, portanto, possuir uma quantidade de parâmetros muito superior à quantidade de parâmetros simultaneamente residentes ou executados.

A proposta nasce da convergência de duas arquiteturas experimentais anteriores.

A primeira, **Eyle Memory Graph**, demonstrou uma separação entre capacidade total de memória e quantidade de memória efetivamente materializada no contexto cognitivo.

A segunda, **MiniMaxBrain (MMB)**, explorou uma separação equivalente no nível físico: o tamanho lógico de um modelo pode ser superior à quantidade de parâmetros simultaneamente residentes em RAM, desde que parâmetros possam ser endereçados, carregados, reutilizados e descartados de forma eficiente.

A GSNA generaliza os dois princípios:

> **O tamanho total de um sistema inteligente não deveria determinar diretamente o custo de cada operação cognitiva.**

O objetivo é investigar modelos cuja capacidade possa crescer através de um grafo de computação e memória enquanto o custo marginal de inferência permaneça associado principalmente ao **subgrafo ativado**, e não ao tamanho total do sistema.

---

# 1. Origem da proposta

A GSNA não surgiu inicialmente de uma tentativa de substituir Transformers.

Ela surgiu de uma investigação sobre memória.

## 1.1 Eyle: memória cognitiva endereçável

A arquitetura Memory Graph do Eyle parte de uma propriedade fundamental:

> crescimento da memória persistente não implica crescimento automático do contexto fornecido ao modelo.

Um sistema pode possuir milhares ou milhões de elementos armazenados e ainda assim materializar somente os elementos necessários para uma determinada operação.

O Eyle adiciona ainda propriedades epistemológicas à memória.

Uma unidade de memória pode possuir:

- identidade;
- escopo;
- domínio;
- contexto;
- retenção;
- natureza epistemológica;
- confiança;
- volatilidade;
- temporalidade;
- relações;
- proveniência;
- revisão;
- estado de freshness;
- histórico de mudanças.

Consequentemente, memória deixa de ser apenas texto recuperado por similaridade.

Ela se torna **estado persistente estruturado**.

Isso levou a uma pergunta mais ampla:

> Se memória pode ser enorme sem estar integralmente presente no contexto, por que os parâmetros de uma rede precisam estar integralmente presentes na memória física?

---

# 2. MiniMaxBrain: memória física endereçável

O MiniMaxBrain explora essa segunda pergunta no contexto de modelos existentes.

Sua arquitetura trata partes dos parâmetros de modelos Mixture-of-Experts como recursos que podem possuir estados físicos diferentes.

Conceitualmente:

```text
Modelo lógico
      │
      ├── parâmetros residentes
      │
      └── parâmetros não residentes
                  │
                  ▼
            Parameter Store
                  │
                  ▼
                SSD
```

Quando determinado expert é necessário:

```text
router
   │
   ▼
expert ID
   │
   ▼
pager/cache
   │
   ▼
parâmetros
   │
   ▼
compute
```

Isso estabelece uma segunda propriedade:

> **tamanho lógico do modelo não precisa ser equivalente ao seu working set físico.**

O MMB, entretanto, opera sobre modelos que não foram originalmente treinados para esse paradigma.

O roteamento, as camadas e os padrões de acesso continuam determinados pela arquitetura original.

Portanto, paging precisa adaptar-se ao modelo.

A GSNA inverte essa relação.

> **O modelo deve ser treinado desde o início para ser paginável.**

---

# 3. Hipótese central

Considere um modelo tradicional como uma função:

\[
y = F_\theta(x)
\]

onde \(\theta\) representa um grande conjunto relativamente estático de parâmetros.

Na GSNA, o modelo é representado como um grafo:

\[
G = (V,E)
\]

Cada vértice pode representar uma unidade computacional, memória ou mecanismo de controle.

Um nó computacional \(v_i\) possui parâmetros:

\[
\theta_i
\]

e implementa uma transformação:

\[
h_{k+1}=f_i(h_k;\theta_i)
\]

Um roteador determina o próximo conjunto de nós:

\[
V_{k+1}^{active}
=
R(h_k,S_k,G)
\]

onde:

- \(h_k\) é o estado neural atual;
- \(S_k\) é o estado persistente relevante;
- \(G\) representa a estrutura endereçável do modelo.

A computação de uma entrada deixa de necessariamente percorrer todas as camadas.

Ela percorre um **caminho dinâmico pelo grafo**.

---

# 4. Invariantes arquiteturais

A GSNA é construída em torno de cinco invariantes.

### 4.1 Parâmetros totais ≠ parâmetros residentes

\[
|\Theta_{total}| \neq |\Theta_{resident}|
\]

Um modelo pode possuir uma grande quantidade de parâmetros enquanto somente uma fração permanece em RAM ou VRAM.

---

### 4.2 Parâmetros totais ≠ parâmetros ativos

\[
|\Theta_{total}| \neq |\Theta_{active}|
\]

Somente os nós selecionados pelo roteamento participam da computação corrente.

---

### 4.3 Memória total ≠ contexto ativo

\[
|M_{total}| \neq |M_{active}|
\]

Memória persistente não é automaticamente transformada em contexto.

---

### 4.4 Histórico total ≠ estado recorrente

O crescimento do histórico de interação não deve exigir crescimento proporcional do estado neural imediato.

---

### 4.5 Complexidade da tarefa ≠ profundidade fixa

Uma operação simples não deve necessariamente executar a mesma quantidade de computação de uma operação difícil.

---

# 5. A rede como grafo

Em vez de:

```text
Layer 1
   ↓
Layer 2
   ↓
Layer 3
   ↓
...
   ↓
Layer N
```

a GSNA propõe:

```text
                 Node 17
                /       \
               /         \
          Node 81       Node 403
              \          /
               \        /
                Node 92
                   │
              Router State
               /       \
              /         \
        Node 140       Node 9001
```

Para determinada operação, o caminho poderia ser:

```text
17 → 81 → 92 → 140
```

Enquanto outra entrada poderia produzir:

```text
17 → 403 → 92 → 9001 → 722 → 140
```

Não existe obrigação de executar todos os nós.

---

# 6. Tipos fundamentais de nós

A primeira especificação da GSNA define quatro famílias.

## 6.1 Compute Nodes

Contêm parâmetros responsáveis por transformações neurais.

\[
h' = f_i(h;\theta_i)
\]

Podem desenvolver especializações durante treinamento:

- linguagem;
- código;
- matemática;
- entidades;
- sintaxe;
- planejamento;
- abstração;
- raciocínio;
- multimodalidade.

Essas especializações não precisam ser definidas manualmente.

---

## 6.2 Router Nodes

Decidem qual região do grafo deve ser ativada.

Recebem:

\[
R(h,S,M)
\]

e produzem uma distribuição esparsa:

\[
P(V_{next}|h,S)
\]

Exemplo:

```text
node 812    0.46
node 817    0.27
node 1092   0.14
node 31     0.08
```

O runtime pode utilizar essa distribuição tanto para execução quanto para prefetch.

---

## 6.3 Neural Memory Nodes

Representam memória distribuída e diferenciável.

Servem para informações difíceis de representar como fatos discretos:

- padrões;
- representações comprimidas;
- associações;
- hábitos;
- regularidades;
- features aprendidas.

São atualizados por mecanismos neurais.

---

## 6.4 Epistemic Memory Nodes

Representam memória explícita.

Herdam conceitos desenvolvidos no Eyle:

```text
identity
nature
confidence
volatility
temporal validity
scope
relations
provenance
revision
freshness
```

Esses nós não existem para substituir parâmetros.

Eles representam conhecimento mutável, episódico ou verificável.

---

# 7. Três escalas de memória

A GSNA distingue três regimes.

## Fast State

Estado utilizado durante processamento imediato:

\[
H_t
\]

Possui baixo custo e alta frequência de atualização.

---

## Neural Memory

Estado intermediário:

\[
N_t
\]

Armazena representações persistentes ou semipersistentes aprendidas.

---

## Epistemic Memory

Grafo explícito:

\[
M_t
\]

Contém conhecimento com identidade e proveniência.

O estado cognitivo completo pode ser representado por:

\[
S_t=(H_t,N_t,M_t)
\]

---

# 8. Computação adaptativa

Um dos objetivos centrais é eliminar profundidade obrigatoriamente fixa.

Considere:

\[
h_{k+1}=f_{v_k}(h_k)
\]

Depois de cada transição, um controlador estima:

\[
P(stop|h_k)
\]

Quando:

\[
P(stop|h_k)>\tau
\]

a computação pode terminar.

Consequentemente:

```text
entrada simples
      ↓
3–5 passos
      ↓
output
```

enquanto:

```text
problema complexo
      ↓
20–50 passos
      ↓
output
```

O mesmo modelo pode oferecer diferentes budgets:

```text
low-compute
balanced
deep-compute
```

sem necessidade de modelos diferentes.

---

# 9. Parameter Graph

Os parâmetros deixam de ser vistos como uma única estrutura monolítica.

Temos:

\[
\Theta =
\{\theta_1,\theta_2,\ldots,\theta_n\}
\]

associados a nós endereçáveis.

Cada bloco pode possuir:

```text
node_id
parameter_location
size
dependencies
routing_metadata
residency_state
usage_statistics
```

O runtime mantém apenas um working set:

\[
W_t \subset \Theta
\]

com:

\[
|W_t| \ll |\Theta|
\]

quando a sparsity for suficientemente alta.

---

# 10. Parameter Paging

Quando um nó selecionado não está residente:

```text
Graph Router
     │
     ▼
 Node ID
     │
     ▼
Residency Manager
     │
 ┌───┴──────────────┐
 │                  │
resident        non-resident
 │                  │
 ▼                  ▼
compute         Parameter Store
                    │
                    ▼
                   SSD
                    │
                    ▼
                  cache
                    │
                    ▼
                  compute
```

O conceito desenvolvido no MMB torna-se parte da especificação da rede.

O paging não é um fallback.

É um componente arquitetural.

---

# 11. Predictive Prefetch

O roteador não precisa produzir somente o próximo nó.

Pode produzir candidatos futuros:

\[
P(v_{k+1}|h_k)
\]

Isso permite que o runtime faça:

```text
execute node 100
      │
      ├── prefetch 812
      └── prefetch 817
```

durante a computação atual.

Assim:

\[
T_{IO}
\]

pode ser parcialmente ocultado por:

\[
T_{compute}
\]

O treinamento pode explicitamente recompensar previsibilidade de caminho.

---

# 12. Localidade como propriedade aprendida

Arquiteturas convencionais otimizam principalmente a qualidade da previsão.

A GSNA introduz custo físico na função objetivo.

Conceitualmente:

\[
L =
L_{language}
+\lambda_r L_{routing}
+\lambda_c L_{compute}
+\lambda_l L_{locality}
+\lambda_m L_{memory}
+\lambda_p L_{paging}
\]

### Language loss

Preserva capacidade linguística.

### Routing loss

Treina seleção eficiente de nós.

### Compute loss

Penaliza computação desnecessária.

### Locality loss

Favorece trajetórias que reutilizem regiões já residentes quando não houver perda semântica significativa.

### Memory loss

Treina operações corretas sobre memória.

### Paging loss

Penaliza padrões de acesso fisicamente caros.

O modelo passa, portanto, a aprender parcialmente **como executar eficientemente no hardware alvo**.

---

# 13. Quatro formas de sparsity

A GSNA combina quatro propriedades.

| Forma | Objetivo |
|---|---|
| Parameter sparsity | Ativar pequena fração dos parâmetros |
| Compute sparsity | Executar apenas quantidade necessária de passos |
| Memory sparsity | Recuperar apenas memória relevante |
| Residency sparsity | Manter pequena fração dos parâmetros em RAM/VRAM |

O objetivo não é maximizar uma delas isoladamente.

É encontrar um equilíbrio que preserve qualidade.

---

# 14. Separação entre Parameter Graph e Memory Graph

É fundamental não confundir dois grafos.

## Parameter Graph

Representa **capacidade aprendida**.

Contém:

```text
compute nodes
routers
neural representations
parameter topology
```

Muda principalmente durante treinamento.

---

## Memory Graph

Representa **experiência e conhecimento mutável**.

Contém:

```text
facts
observations
hypotheses
preferences
relations
events
provenance
```

Pode mudar durante inferência.

Assim:

\[
G_P = ParameterGraph
\]

\[
G_M = MemoryGraph
\]

O estado completo passa a ser:

\[
System =
(G_P,G_M,H_t)
\]

---

# 15. Uma máquina neural paginada

A consequência arquitetural é que o modelo deixa de ser apenas um arquivo carregado para RAM.

Ele passa a possuir uma hierarquia física:

```text
             COMPUTE
                ▲
                │
          active parameters
                │
             VRAM/L3
                ▲
                │
             RAM cache
                ▲
                │
          Parameter Store
                ▲
                │
               SSD
```

A inteligência do runtime está em manter:

\[
P(parameter\ needed\ soon | state)
\]

alto para aquilo que permanece próximo ao compute.

Esse problema passa a ser parcialmente neural e parcialmente sistêmico.

---

# 16. Hardware-aware intelligence

Uma característica potencialmente nova da GSNA é que hardware deixa de ser apenas plataforma de execução.

Ele pode fornecer sinais para treinamento.

Exemplos:

```text
page faults
cache hits
bytes loaded
latency
active parameters
FLOPs
memory pressure
prefetch accuracy
```

O treinamento pode incorporar esses sinais.

Duas trajetórias com qualidade linguística equivalente podem receber rewards diferentes:

```text
Trajectory A
quality = .95
300 MB carregados

Trajectory B
quality = .95
4 GB carregados
```

A trajetória A deve ser preferida.

---

# 17. Crescimento estrutural

Uma possibilidade de pesquisa de longo prazo é permitir expansão do Parameter Graph.

Em vez de definir permanentemente:

\[
|\Theta| = constante
\]

poderíamos permitir:

\[
G_P^{t+1}
=
Grow(G_P^t)
\]

quando determinados critérios forem satisfeitos.

Por exemplo:

```text
1B
 ↓
specialization pressure
 ↓
novo cluster
 ↓
1.2B
 ↓
novas competências
 ↓
2B
```

Isso permitiria estudar aprendizado estrutural sem exigir que toda nova capacidade modifique toda a rede.

Essa característica permanece uma hipótese de pesquisa e não é requisito para a primeira implementação.

---

# 18. O que não está sendo proposto

A GSNA não é simplesmente:

```text
Transformer + RAG
```

nem:

```text
Transformer + MoE
```

nem:

```text
Mamba + SQLite
```

nem:

```text
GNN para processamento de texto
```

A hipótese é mais profunda.

O **grafo define a própria organização da computação neural**.

Memória, parâmetros e computação tornam-se recursos endereçáveis.

---

# 19. Objetivo principal

A meta pode ser expressa da seguinte forma:

> Construir um modelo de linguagem no qual o custo marginal de inferência seja determinado predominantemente pela complexidade da operação e pelo subgrafo ativado, e não diretamente pelo tamanho total do modelo, da memória persistente ou do histórico acumulado.

Idealmente:

\[
Cost(token)
\approx
Cost(active\ subgraph)
\]

e não:

\[
Cost(token)
\approx
Cost(total\ model)
\]

---

# 20. Objetivos mensuráveis

A arquitetura deve ser avaliada simultaneamente em qualidade e eficiência.

## Qualidade

- perplexidade;
- reasoning;
- factualidade;
- coding;
- long-term memory;
- continual interaction.

## Computação

\[
FLOPs/token
\]

## Ativação

\[
\frac{active\ parameters}{total\ parameters}
\]

## Residência

\[
\frac{resident\ parameters}{total\ parameters}
\]

## I/O

\[
bytes/token
\]

## Cache

\[
cache\ hit\ rate
\]

## Roteamento

\[
routing\ entropy
\]

e distribuição de utilização dos nós.

## Memória

- precision de recall;
- recall útil;
- escrita desnecessária;
- atualização correta;
- preservação de proveniência.

---

# 21. Critérios de sucesso

O primeiro objetivo não é superar grandes modelos comerciais.

É demonstrar a hipótese estrutural.

Uma primeira prova forte seria:

> Um GSNA pequeno alcança qualidade próxima de um baseline denso de tamanho equivalente utilizando significativamente menos parâmetros ativos por token.

A segunda:

> O modelo mantém qualidade quando grande parte dos parâmetros deixa de estar residente.

A terceira:

> O custo de inferência permanece relativamente estável quando a capacidade total do Parameter Graph aumenta.

A quarta:

> Memória persistente pode crescer várias ordens de magnitude sem crescimento proporcional do estado cognitivo ativo.

---

# 22. Programa experimental

A pesquisa deve ser incremental.

## Fase 0 — Formalização

Definir:

- tipos de nós;
- representação de estado;
- função de roteamento;
- operação de cada nó;
- mecanismo de parada;
- função de loss;
- formato físico do Parameter Graph.

Nenhuma otimização prematura de kernel.

---

## Fase 1 — Graph Language Model mínimo

Construir aproximadamente:

```text
10M–50M parâmetros
```

Inteiramente residente em memória.

Objetivo:

> provar que roteamento dinâmico consegue aprender linguagem.

Comparar com Transformer denso de escala semelhante.

---

## Fase 2 — Sparse Parameter Graph

Expandir para:

```text
50M–200M parâmetros totais
```

com pequena fração ativa.

Medir:

```text
quality
active parameters
routing stability
specialization
FLOPs
```

Nenhum paging ainda.

---

## Fase 3 — Adaptive Compute

Adicionar mecanismo de parada.

Medir distribuição:

```text
steps/token
```

e verificar se tarefas diferentes desenvolvem profundidades diferentes.

---

## Fase 4 — Paged Parameter Graph

Integrar conceitos do MiniMaxBrain.

Adicionar:

```text
resident set
parameter store
cache
leases
prefetch
eviction
```

Agora testar:

\[
resident \ll total
\]

---

## Fase 5 — Hardware-aware routing

Introduzir:

```text
locality loss
paging penalty
cache reward
prefetch prediction
```

Verificar se a própria rede aprende trajetórias mais favoráveis ao hardware.

---

## Fase 6 — Epistemic Memory

Integrar os princípios do Eyle.

A rede passa a aprender:

```text
RECALL
REMEMBER
REVISE
RELATE
SUPERSEDE
IGNORE
```

A memória permanece externa ao Parameter Graph, mas faz parte do estado cognitivo.

---

## Fase 7 — Long-horizon learning

Criar ambientes com dezenas ou centenas de milhares de eventos.

Avaliar:

- persistência;
- contradições;
- mudanças temporais;
- atualização;
- provenance;
- esquecimento;
- consolidação.

---

## Fase 8 — Structural Growth

Somente após as etapas anteriores:

> investigar criação, divisão, fusão e aposentadoria de nós computacionais durante treinamento.

---

# 23. Primeiro protótipo

A primeira implementação deve deliberadamente evitar escala.

Uma configuração plausível:

```text
hidden dimension: 512
graph nodes: 256–1024
active nodes/step: 2–8
max graph steps: 16
parameters: 20M–50M
```

Treinamento inicial:

```text
next-token prediction
+
routing regularization
+
compute penalty
+
load balancing
```

A pergunta inicial é somente:

> **Um grafo neural dinamicamente roteado consegue aprender modelagem autoregressiva de linguagem sem depender de uma pilha Transformer convencional?**

Se a resposta for negativa, devemos descobrir cedo.

Se positiva, todo o restante se torna justificável.

---

# 24. Baselines obrigatórios

Para evitar conclusões falsas, cada experimento deve possuir pelo menos:

```text
Dense Transformer
MoE Transformer
recorrente/SSM moderno
GSNA
```

com orçamento comparável de treinamento.

Não basta comparar parâmetros totais.

Devemos comparar:

\[
training\ FLOPs
\]

\[
inference\ FLOPs
\]

\[
active\ parameters
\]

\[
resident\ bytes
\]

\[
storage\ bandwidth
\]

\[
latency
\]

\[
quality
\]

---

# 25. Riscos científicos

## Routing collapse

O modelo pode aprender a utilizar sempre os mesmos nós.

## Excessive fragmentation

Especialização excessiva pode produzir milhares de nós pouco úteis.

## I/O domination

Sparsity computacional não garante velocidade se o sistema realizar muitos acessos aleatórios ao armazenamento.

## Training instability

Roteamento discreto e topologia dinâmica dificultam otimização.

## Catastrophic routing changes

Pequenas atualizações podem alterar trajetórias inteiras.

## Graph overhead

O mecanismo de roteamento não pode custar mais do que a computação economizada.

## Memory pollution

Memória persistente sem políticas adequadas pode acumular conhecimento redundante ou incorreto.

## Hardware overfitting

Treinar agressivamente para determinada hierarquia de cache pode reduzir portabilidade.

Esses riscos são parte central do programa experimental.

---

# 26. Princípio de co-design

A GSNA assume que arquitetura neural e runtime não devem ser desenvolvidos independentemente.

O modelo precisa conhecer propriedades abstratas da máquina:

```text
resident
cheap
expensive
local
remote
available
prefetchable
```

E o runtime precisa compreender previsões do modelo:

```text
likely next nodes
expected lifetime
priority
reuse probability
```

Forma-se um ciclo:

```text
Neural Router
     │
     ▼
Runtime
     │
     ▼
Hardware
     │
 telemetry
     ▼
Training objective
     │
     └────────► Neural Router
```

Esse **hardware/model co-design** é um dos pilares da proposta.

---

# 27. Da memória para a arquitetura

A genealogia do projeto pode ser resumida em três perguntas.

### Eyle

> Como uma inteligência pode possuir memória crescente sem colocar toda essa memória no contexto?

Resposta:

> memória endereçável e ativação seletiva.

### MiniMaxBrain

> Se memória cognitiva pode ser paginada, por que parâmetros neurais precisam permanecer integralmente residentes?

Resposta experimental:

> parâmetros também podem ser tratados como working sets endereçáveis.

### GSNA

Isso produz a terceira pergunta:

> Se memória e parâmetros podem ser seletivamente ativados, por que a própria computação precisa seguir uma estrutura neural fixa?

A GSNA nasce dessa pergunta.

---

# 28. A visão

O objetivo de longo prazo não é construir simplesmente uma LLM que utilize menos RAM.

É investigar uma forma diferente de organizar capacidade neural.

Em vez de:

```text
modelo
=
uma enorme função estática
```

propõe-se:

```text
modelo
=
grafo de capacidades
+
estado
+
roteamento
+
memória
+
runtime
```

A capacidade total pode ser grande.

A atividade instantânea permanece pequena.

O conhecimento acumulado pode ser enorme.

A memória ativa permanece seletiva.

Uma tarefa simples utiliza pouco compute.

Uma tarefa difícil pode explorar uma região maior do grafo.

Parâmetros pouco utilizados podem permanecer em armazenamento secundário.

Parâmetros frequentemente utilizados permanecem próximos à computação.

Conhecimento mutável não precisa ser incorporado novamente aos pesos.

E o próprio sistema aprende quais recursos utilizar.

---

# 29. Hipótese final

A hipótese científica da GSNA é:

> **Inteligência neural escalável não exige que toda capacidade aprendida esteja simultaneamente ativa, residente ou presente no contexto.**

Se capacidade computacional, conhecimento e estado puderem ser representados como recursos endereçáveis dentro de um grafo dinamicamente roteado, torna-se possível investigar modelos cujo tamanho lógico cresça muito mais rapidamente que seu working set físico.

A consequência potencial é uma nova relação entre:

\[
capacidade
\]

\[
memória
\]

\[
compute
\]

e:

\[
hardware
\]

onde esses quatro valores deixam de crescer necessariamente juntos.

---

# 30. Princípio fundador

Eyle estabeleceu:

> **Memory size ≠ Context size.**

MiniMaxBrain estendeu:

> **Model size ≠ Resident model size.**

A Graph-State Neural Architecture propõe a generalização:

> **System capacity ≠ Active computation.**

Essa é a hipótese que o projeto deve tentar provar.