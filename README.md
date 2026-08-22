# Graph-State Neural Architecture

## Uma arquitetura neural de microcore, grafo de capacidades, memória hierárquica, computação esparsa e parâmetros endereçáveis

**White Paper — Versão 0.2**  
**Status:** Proposta arquitetural, hipótese científica e programa experimental

---

## Resumo

Os modelos neurais modernos são normalmente construídos de forma que uma grande parcela de sua capacidade permaneça próxima ao mecanismo de computação durante a inferência. Mesmo em arquiteturas esparsas, a topologia de execução costuma ser relativamente fixa e o sistema ainda depende de uma quantidade significativa de parâmetros residentes em RAM ou VRAM.

Essa organização cria uma relação forte entre:

- tamanho total do modelo;
- memória física necessária;
- quantidade de compute executada;
- largura de banda exigida;
- custo de treinamento;
- custo de inferência.

A **Graph-State Neural Architecture (GSNA)** propõe investigar uma relação diferente.

Na GSNA, a inteligência não é representada primariamente como um único bloco monolítico de parâmetros. Ela é organizada como:

```text
Neural Microcore
+
Capability Graph
+
Capability Address Space
+
Memory Graph
+
Hierarchical Router
+
Residency Runtime
```

O **Neural Microcore** é uma estrutura pequena e permanentemente residente. Ele mantém estado, produz representações, coordena roteamento, integra resultados, controla profundidade de computação e interage com memória.

A maior parte da capacidade aprendida pode existir em um **Capability Graph** muito maior, composto por módulos neurais endereçáveis. Esses módulos podem residir em diferentes níveis da hierarquia física:

```text
VRAM
RAM
SSD / NVMe
HDD
armazenamento remoto
```

A cada operação, apenas uma pequena região do grafo precisa ser ativada e materializada próxima ao compute.

Consequentemente, a capacidade total do sistema pode ser muito maior que seu working set físico.

A GSNA acrescenta uma hipótese ainda mais forte:

> **O modelo deve ser treinado desde o início para funcionar sob um orçamento físico limitado.**

Memória disponível, largura de banda, latência, número de parâmetros ativos, quantidade de I/O e FLOPs podem participar do treinamento como restrições ou sinais de custo.

Assim, o objetivo não é simplesmente construir uma LLM que utilize menos RAM.

O objetivo é investigar uma arquitetura em que:

\[
SystemCapacity
\gg
ResidentCapacity
\]

enquanto:

\[
Cost(operation)
\approx
Cost(active\ subgraph)
\]

e não:

\[
Cost(operation)
\approx
Cost(total\ system)
\]

Se essa hipótese for validada, computadores modestos poderão acessar sistemas neurais muito maiores do que sua memória local, enquanto servidores e clusters de grande porte poderão utilizar o mesmo princípio para operar capacidades lógicas muito superiores às que caberiam integralmente em memória de alta velocidade.

O efeito potencial é alterar a relação entre **capacidade neural, hardware, custo e acessibilidade**.

---

# 1. O problema

O crescimento de modelos neurais tem sido acompanhado por crescimento de:

```text
parâmetros
RAM
VRAM
bandwidth
compute
energia
infraestrutura
```

Mesmo quando apenas parte dos parâmetros é ativada por token, grandes sistemas frequentemente continuam exigindo uma base residente substancial.

Isso produz dois limites.

## 1.1 Limite de acessibilidade

Um modelo pode existir publicamente e ainda ser impraticável para a maioria das máquinas.

Se o modelo exige dezenas ou centenas de GB de memória rápida, seu acesso fica restrito a:

- servidores;
- estações especializadas;
- clusters;
- GPUs de alto custo.

## 1.2 Limite de escala dos grandes sistemas

O problema também existe no outro extremo.

Uma organização que já possui centenas ou milhares de GPUs ainda precisa decidir:

> quanto da capacidade do modelo deve permanecer simultaneamente residente?

Adicionar parâmetros normalmente exige adicionar memória, interconexão, compute ou ambos.

Portanto, a questão não é apenas:

> Como fazer modelos grandes caberem em computadores pequenos?

A questão mais geral é:

> **Como permitir que a capacidade lógica de um sistema neural cresça muito mais rapidamente que seu working set físico?**

---

# 2. Hipótese central

A hipótese científica da GSNA é:

> **Inteligência neural escalável não exige que toda capacidade aprendida esteja simultaneamente ativa, residente, presente no contexto ou envolvida em cada operação.**

Formalmente, considere:

\[
\Theta_{total}
\]

como todos os parâmetros do sistema.

A GSNA busca operar com:

\[
\Theta_{active}(t) \subset \Theta_{resident}(t) \subset \Theta_{total}
\]

e idealmente:

\[
|\Theta_{active}(t)|
\ll
|\Theta_{resident}(t)|
\ll
|\Theta_{total}|
\]

A capacidade total pode crescer.

A atividade instantânea permanece pequena.

---

# 3. Princípios fundamentais

A GSNA é construída em torno das seguintes separações.

## 3.1 Capacidade total ≠ capacidade residente

\[
SystemCapacity
\neq
ResidentCapacity
\]

## 3.2 Parâmetros totais ≠ parâmetros ativos

\[
|\Theta_{total}|
\neq
|\Theta_{active}|
\]

## 3.3 Memória persistente ≠ contexto ativo

\[
|M_{total}|
\neq
|M_{active}|
\]

## 3.4 Histórico total ≠ estado neural imediato

O crescimento de experiência não deve exigir estado recorrente proporcionalmente maior.

## 3.5 Complexidade da tarefa ≠ profundidade fixa

Tarefas diferentes podem consumir quantidades diferentes de compute.

## 3.6 Estado total de treinamento ≠ estado residente de treinamento

\[
TrainingState_{total}
\neq
TrainingState_{resident}
\]

## 3.7 Capacidade total ≠ bytes transferidos por token

\[
CapabilitySize
\neq
BytesFetched/token
\]

## 3.8 Índice total ≠ índice residente

\[
Index_{total}
\neq
Index_{resident}
\]

O catálogo de capacidades também deve poder crescer sem permanecer integralmente em RAM.

## 3.9 Capacidade lógica ≠ endereço físico

\[
CapabilityID
\neq
PhysicalAddress
\]

O modelo seleciona capacidades lógicas.

O runtime resolve onde seus parâmetros estão fisicamente.

---

# 4. Visão geral da arquitetura

A GSNA pode ser representada como:

```text
                         INPUT
                           │
                           ▼
                ┌────────────────────┐
                │  NEURAL MICROCORE  │
                │                    │
                │ state              │
                │ routing query      │
                │ integration        │
                │ stop/continue      │
                │ resource policy    │
                └─────────┬──────────┘
                          │
                    semantic query
                          │
                          ▼
                ┌────────────────────┐
                │ CAPABILITY ADDRESS │
                │       SPACE        │
                └─────────┬──────────┘
                          │
                    capability IDs
                          │
                          ▼
                ┌────────────────────┐
                │ CAPABILITY GRAPH   │
                │                    │
                │ node 17            │
                │ node 812           │
                │ node 9001          │
                │ ...                │
                └─────────┬──────────┘
                          │
                    required pages
                          │
                          ▼
                ┌────────────────────┐
                │ RESIDENCY RUNTIME  │
                │ cache / pager      │
                │ prefetch / leases  │
                └─────────┬──────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           VRAM          RAM         STORAGE
                                        │
                                        ▼
                                 GB / TB / beyond
```

Separadamente:

```text
Neural Microcore
       │
       ▼
Memory Controller
       │
       ▼
Memory Graph
```

Parâmetros e memória persistente são recursos diferentes.

---

# 5. Neural Microcore

O **Neural Microcore** é o núcleo sempre residente da arquitetura.

Ele não deve conter toda a capacidade do sistema.

Sua função principal é coordenar.

## 5.1 Responsabilidades

O microcore pode conter:

- representação do estado atual;
- mecanismo recorrente ou state-space;
- capacidade linguística basal;
- geração de routing queries;
- integração de módulos;
- controlador de parada;
- predição de próximos recursos;
- controlador de memória;
- representação abstrata do orçamento físico.

## 5.2 Objetivo de tamanho

A hipótese de longo prazo é manter o microcore pequeno mesmo quando o Capability Graph cresce.

Idealmente:

\[
|Core|
\approx
constante
\]

enquanto:

\[
|CapabilityGraph|
\rightarrow
grande
\]

Uma meta experimental futura poderia investigar regimes como:

```text
microcore:             centenas de MiB
working cache:         centenas de MiB
total active footprint ~1 GiB
```

enquanto o Capability Graph possui:

```text
10 GB
100 GB
2 TB
20 TB
```

Esses valores são metas de pesquisa, não garantias.

---

# 6. Como um microcore pequeno encontra capacidades enormes

Um microcore pequeno não pode memorizar diretamente a localização de milhões ou bilhões de módulos.

A solução é separar:

```text
saber a informação
```

de:

```text
saber navegar até a informação
```

O microcore produz uma **routing query**:

\[
q_k = Q(h_k)
\]

Essa query representa a necessidade atual.

Exemplo conceitual:

```text
"preciso de capacidade relacionada a
 álgebra simbólica e resolução de equações"
```

A query não precisa ser texto.

Normalmente seria uma representação neural compacta.

Essa query é usada para navegar por um espaço externo de capacidades.

---

# 7. Capability Address Space

A GSNA introduz um **Capability Address Space**.

Seu papel é mapear necessidades semânticas para capacidades lógicas.

```text
state
  │
  ▼
routing query
  │
  ▼
Capability Address Space
  │
  ▼
Capability IDs
```

Uma capacidade pode possuir:

```text
capability_id
routing_key
node_type
input_contract
output_contract
version
dependencies
cost_profile
residency_metadata
```

A localização física não precisa fazer parte da representação neural.

---

# 8. Endereço lógico e endereço físico

A separação é:

```text
Neural layer:
"quero capability 0x02:A71:184"

Runtime:
"está no SSD 2, shard 18, offset X"

ou:
"já está no RAM cache"

ou:
"já está em VRAM"
```

Portanto:

\[
LogicalCapability
\rightarrow
RuntimeResolution
\rightarrow
PhysicalLocation
\]

Isso permite mover, compactar, replicar ou redistribuir parâmetros sem alterar a identidade neural da capacidade.

---

# 9. Índice hierárquico

Se existirem milhões ou bilhões de nodes, não é viável comparar uma query contra todos eles.

A GSNA propõe roteamento hierárquico.

```text
query
 │
 ▼
root
 │
 ├── language
 ├── math
 ├── code
 ├── planning
 └── multimodal
       │
       ▼
   cluster
       │
       ▼
  subcluster
       │
       ▼
 local candidates
       │
       ▼
 capability IDs
```

Formalmente:

\[
c_1 = R_0(q)
\]

\[
c_2 = R_1(q,c_1)
\]

\[
...
\]

\[
v = R_n(q,c_1,\ldots,c_n)
\]

A busca deixa de depender linearmente do número total de módulos.

---

# 10. O próprio índice pode ser paginado

Um índice enorme também consome memória.

Portanto:

```text
microcore
   ↓
resident root
   ↓
paged index region
   ↓
paged sub-index
   ↓
capability
```

Assim:

\[
Index_{resident}
\ll
Index_{total}
\]

O princípio de working set aplica-se recursivamente.

A inteligência mantém perto do compute apenas as regiões do índice necessárias para navegar pelo estado atual.

---

# 11. Capability Graph

O **Capability Graph** representa a capacidade aprendida do sistema.

Formalmente:

\[
G_C=(V,E)
\]

Cada vértice \(v_i\) pode possuir parâmetros:

\[
\theta_i
\]

e executar:

\[
h' = f_i(h;\theta_i)
\]

As arestas podem representar:

- compatibilidade;
- dependência;
- provável sequência;
- especialização;
- composição;
- relações de routing.

O grafo não precisa ser uma sequência fixa de layers.

---

# 12. A computação como trajetória

Uma operação pode percorrer:

```text
17 → 81 → 92 → 140
```

Outra:

```text
17 → 403 → 92 → 9001 → 722 → 140
```

O número de passos também pode variar.

A trajetória é determinada pelo estado, pelo objetivo e pelo orçamento disponível.

---

# 13. Capability Nodes

Capability Nodes representam módulos neurais especializados.

Eles podem aprender funções como:

- linguagem;
- matemática;
- código;
- planejamento;
- abstração;
- percepção;
- domínio científico;
- manipulação simbólica;
- ferramentas;
- multimodalidade.

Essas categorias não precisam ser fixadas manualmente.

Especialização pode emergir do treinamento.

---

# 14. Transformações residuais

Uma forma desejável de node é:

\[
h_{out}
=
h_{in}
+
\Delta_i(h_{in};\theta_i)
\]

Isso permite que o microcore mantenha uma baseline funcional.

Módulos adicionam capacidade incremental.

Vantagens potenciais:

- degradação graciosa;
- atualização modular;
- especialização;
- continual learning;
- menor interferência global;
- expansão incremental.

---

# 15. Tipos de nodes

Uma primeira taxonomia inclui:

## 15.1 Compute Nodes

Transformações neurais gerais.

## 15.2 Capability Nodes

Especializações maiores.

## 15.3 Router Nodes

Selecionam regiões do espaço de capacidades.

## 15.4 Integration Nodes

Combinam múltiplos resultados.

## 15.5 Neural Memory Nodes

Mantêm representações persistentes ou semipersistentes.

## 15.6 Epistemic Memory Nodes

Representam conhecimento explícito e revisável.

---

# 16. Memory Graph

O Memory Graph é separado do Capability Graph.

Ele representa experiência e conhecimento mutável.

Pode conter:

```text
identity
statement
confidence
scope
temporal validity
provenance
relations
revision
freshness
```

Isso permite distinguir:

```text
capacidade aprendida
```

de:

```text
conhecimento mutável
```

O objetivo é evitar que fatos transitórios precisem ser permanentemente incorporados aos pesos.

---

# 17. Três escalas de estado e memória

A GSNA distingue:

## Fast State

\[
H_t
\]

Estado imediato.

## Neural Memory

\[
N_t
\]

Representações persistentes aprendidas.

## Epistemic Memory

\[
M_t
\]

Conhecimento explícito e revisável.

O estado cognitivo pode ser representado como:

\[
S_t=(H_t,N_t,M_t)
\]

---

# 18. Parameter Paging como parte da arquitetura

Quando uma capacidade selecionada não está residente:

```text
Capability ID
     │
     ▼
Residency Manager
     │
 ┌───┴───────────┐
 │               │
resident     non-resident
 │               │
 ▼               ▼
compute       Parameter Store
                   │
                   ▼
                 cache
                   │
                   ▼
                compute
```

Paging não é um fallback.

É uma propriedade prevista pela arquitetura.

---

# 19. Hierarquia física

Os parâmetros podem existir em múltiplos níveis:

```text
                 COMPUTE
                    ▲
                    │
                  VRAM
                    ▲
                    │
                 RAM cache
                    ▲
                    │
                SSD / NVMe
                    ▲
                    │
                   HDD
                    ▲
                    │
            armazenamento remoto
```

O runtime decide:

- o que promover;
- o que manter;
- o que evictar;
- o que prefetchar;
- o que replicar.

---

# 20. Residency como working set

A cada instante:

\[
W_t \subset \Theta_{total}
\]

onde \(W_t\) representa os parâmetros residentes.

A meta é:

\[
|W_t|
\ll
|\Theta_{total}|
\]

sem comprometer excessivamente qualidade ou latência.

O tamanho total do Capability Graph deixa de ser equivalente à memória física necessária.

---

# 21. Predictive Prefetch

O sistema pode prever futuras trajetórias:

\[
P(v_{k+1},v_{k+2},...,v_{k+n}\mid h_k)
\]

Isso significa:

> dado o estado atual, quais caminhos futuros parecem mais prováveis?

Exemplo:

```text
compute node 81
      │
      ├── prefetch node 92
      ├── prefetch node 140
      └── manter node 318 em baixa prioridade
```

A predição não substitui o routing real.

Ela apenas prepara o runtime.

```text
prediction
≠
execution decision
```

---

# 22. Prefetch como tarefa aprendida

Durante treinamento, a trajetória real é conhecida.

Se:

```text
predito:
81 → 92 → 140 → 318

real:
81 → 92 → 140 → 722
```

o sistema pode receber uma loss específica.

Conceitualmente:

\[
L_{prefetch}
=
-\sum_j
\log P(v_{k+j}^{real}|h_k)
\]

Assim o modelo aprende não apenas a computar.

Ele aprende a tornar seu próprio caminho previsível.

---

# 23. Localidade como propriedade aprendida

Duas trajetórias podem possuir qualidade semelhante:

```text
Trajectory A
quality = 0.95
100 MB transferidos

Trajectory B
quality = 0.95
2 GB transferidos
```

A arquitetura deve preferir A.

Portanto, localidade deixa de ser apenas otimização de runtime.

Ela participa do treinamento.

---

# 24. A métrica física central: bytes por operação

Ativar poucos parâmetros não garante eficiência.

Um modelo pode ser extremamente esparso e ainda ler grandes volumes do SSD.

Por isso:

\[
\boxed{BytesFetched/token}
\]

é uma métrica de primeira classe.

Também devem ser medidos:

```text
blocking_bytes/token
prefetched_bytes/token
wasted_prefetch_bytes/token
cache_hits
cache_misses
page_faults
reuse_distance
```

A meta é:

> **mover pouco dado para produzir muito trabalho útil.**

---

# 25. Computação adaptativa

A profundidade não precisa ser fixa.

Após cada transição:

\[
P(stop|h_k)
\]

pode ser estimado.

Quando:

\[
P(stop|h_k)>\tau
\]

a operação pode encerrar.

Exemplo:

```text
tarefa simples
   ↓
3 passos
   ↓
output
```

versus:

```text
tarefa complexa
   ↓
30 passos
   ↓
output
```

O mesmo sistema pode operar em:

```text
low-compute
balanced
deep-compute
```

---

# 26. Budget-Constrained Neural Training

A GSNA propõe que o treinamento conheça limites físicos.

Em vez de apenas:

\[
\min L_{quality}
\]

temos qualidade sob restrições:

\[
ResidentBytes \le B
\]

\[
ActiveParameters/token \le A
\]

\[
IOBytes/token \le I
\]

\[
FLOPs/token \le F
\]

\[
Latency \le T
\]

O orçamento torna-se parte do problema de aprendizado.

---

# 27. Função objetivo

Uma forma conceitual é:

\[
L =
L_{task}
+\lambda_r L_{routing}
+\lambda_c L_{compute}
+\lambda_l L_{locality}
+\lambda_m L_{memory}
+\lambda_p L_{paging}
+\lambda_b L_{budget}
+\lambda_i L_{io}
+\lambda_s L_{steps}
+\lambda_u L_{utilization}
+\lambda_f L_{prefetch}
\]

Cada componente mede uma propriedade diferente.

O objetivo não é minimizar custo a qualquer preço.

É encontrar trajetórias eficientes **sem perder capacidade útil**.

---

# 28. Constraints duras

Alguns limites podem ser estruturais:

```text
resident_bytes <= budget
active_nodes <= limit
max_graph_steps <= limit
prefetch_bytes <= budget
```

Uma trajetória que viola o orçamento pode ser inválida.

Isso força o modelo a aprender estratégias executáveis em condições reais.

---

# 29. Resource Dropout

Durante treinamento, o hardware disponível pode variar artificialmente.

Exemplo:

```text
batch A:
RAM = 512 MiB
storage = rápido

batch B:
RAM = 2 GiB
storage = médio

batch C:
RAM = 16 GiB
VRAM = disponível

batch D:
RAM = 1 GiB
storage = lento
```

O microcore recebe um perfil abstrato:

\[
H =
(memory,\ bandwidth,\ latency,\ compute,\ cache)
\]

O routing passa a depender de:

\[
R(h,S,H)
\]

O mesmo modelo pode aprender estratégias diferentes para diferentes máquinas.

---

# 30. Um modelo, múltiplos budgets

Uma consequência desejada é que o mesmo sistema possa operar em:

```text
1 GiB
4 GiB
16 GiB
64 GiB
```

sem exigir checkpoints completamente diferentes.

Com mais hardware, o sistema pode:

- manter mais nodes residentes;
- usar refinements maiores;
- executar trajetórias mais profundas;
- prefetchar mais;
- reduzir latência.

Com menos hardware, pode degradar de forma controlada.

---

# 31. Módulos multirresolução

Uma capacidade pode possuir múltiplos níveis:

\[
\theta_i =
\theta_i^{base}
+
\Delta_i^1
+
\Delta_i^2
+
\Delta_i^3
\]

Exemplo:

```text
base compacta
+ refinement 1
+ refinement 2
+ high precision residual
```

Máquinas pequenas utilizam apenas a base.

Máquinas maiores podem carregar refinements adicionais.

O mesmo Capability Graph passa a oferecer diferentes pontos de qualidade e custo.

---

# 32. Sparse Training State

Treinar um sistema enorme normalmente exige:

```text
weights
gradients
optimizer state
activations
```

A GSNA propõe que apenas módulos utilizados por um batch materializem seu estado de treinamento.

Para node A:

```text
load weights(A)
load optimizer(A)

forward
backward
update

write A
evict
```

Nodes não utilizados permanecem fora da memória.

Assim:

\[
TrainingState_{resident}
\ll
TrainingState_{total}
\]

quando o treinamento for suficientemente esparso.

---

# 33. Page-Major Training

Durante treinamento, requisições podem ser reagrupadas por módulo.

Suponha:

```text
sample 1 → node 81
sample 2 → node 712
sample 3 → node 81
sample 4 → node 81
sample 5 → node 712
```

Em vez de alternar I/O:

```text
81
712
81
81
712
```

o runtime pode fazer:

```text
LOAD 81
process 1,3,4
backward
update
EVICT

LOAD 712
process 2,5
backward
update
EVICT
```

Isso pode transformar drasticamente o padrão de armazenamento durante treinamento.

---

# 34. Crescimento estrutural

A topologia do Capability Graph pode, futuramente, deixar de ser completamente fixa.

O objeto de otimização passa de:

\[
Optimize(\Theta)
\]

para:

\[
Optimize(G,\Theta,R)
\]

O sistema pode investigar operações como:

```text
spawn
split
merge
prune
freeze
retire
```

---

# 35. Split de capacidades

Um node sobrecarregado por funções incompatíveis pode se dividir.

```text
node 31
   │
specialization pressure
   │
   ├── node 31a
   └── node 31b
```

Isso permite que a arquitetura aumente capacidade onde existe demanda real.

---

# 36. Merge e prune

Nodes altamente redundantes podem ser fundidos.

Nodes sem utilidade podem ser aposentados.

A meta é evitar crescimento sem controle.

Capacidade maior só é útil se contribuir para qualidade ou eficiência.

---

# 37. Continual Learning modular

Uma arquitetura modular pode permitir:

```text
core estável

+ capability A
+ capability B
+ capability C
```

sem reescrever necessariamente todo o sistema.

Isso pode ser útil para:

- novas linguagens;
- novos domínios;
- novas ferramentas;
- novas modalidades;
- mudanças científicas;
- especializações privadas.

---

# 38. Layout físico orientado pelo grafo

O runtime pode aprender estatísticas de acesso.

Se:

```text
A → B → C
```

ocorre frequentemente, essas capacidades podem ser colocadas fisicamente próximas.

Isso permite:

- menos random I/O;
- shards mais coerentes;
- melhor read-ahead;
- maior prefetch accuracy;
- menor latência.

Uma fase futura pode executar **graph packing**.

---

# 39. Compilação de trajetórias quentes

Trajetórias frequentes também podem ser otimizadas.

Exemplo:

```text
A → B → C → D
```

pode produzir:

```text
packed storage region
prefetch schedule
fused execution plan
hot residency class
```

A topologia lógica permanece dinâmica.

O runtime otimiza os caminhos mais utilizados.

---

# 40. Arquitetura neural e runtime como um único sistema

A GSNA assume co-design.

```text
Neural Microcore
      │
      ▼
routing / future prediction
      │
      ▼
Runtime
      │
      ▼
Hardware
      │
   telemetry
      ▼
Training Objective
      │
      └────────► Microcore / Router
```

O modelo aprende a utilizar recursos.

O runtime materializa os recursos.

O hardware fornece sinais.

---

# 41. Hardware-aware intelligence

Sinais possíveis:

```text
resident bytes
cache hits
cache misses
page faults
bytes loaded
latency
bandwidth
compute rate
memory pressure
prefetch accuracy
```

Duas trajetórias semanticamente equivalentes podem receber custos diferentes.

O sistema passa a aprender parcialmente:

> **como pensar de forma fisicamente eficiente.**

---

# 42. Portabilidade

A arquitetura não deve aprender detalhes acidentais de uma única máquina.

O treinamento deve trabalhar com propriedades abstratas:

```text
cheap
expensive
resident
remote
fast
slow
prefetchable
reusable
```

O runtime traduz:

```text
"remote + slow"
```

para:

```text
SSD
HDD
network store
```

dependendo da plataforma.

---

# 43. O que a GSNA não é

GSNA não é simplesmente:

```text
Transformer + RAG
```

Não é apenas:

```text
Transformer + MoE
```

Não é apenas:

```text
virtual memory aplicada a weights
```

Não é apenas:

```text
GNN para linguagem
```

A proposta é mais ampla:

> **a organização da computação neural, dos parâmetros, do endereçamento, da memória e do hardware faz parte da própria arquitetura.**

---

# 44. Relação com MoE

MoE tradicional oferece sparsity de experts.

GSNA generaliza o princípio para:

- topologia global;
- profundidade variável;
- routing hierárquico;
- parâmetros não residentes;
- índice paginável;
- memória persistente;
- prefetch treinável;
- budget-aware execution;
- crescimento estrutural.

---

# 45. Relação com sistemas de memória virtual

Existe uma analogia:

```text
Sistema operacional:
virtual address
page table
page cache
scheduler

GSNA:
Capability ID
Capability Address Space
residency cache
neural router
```

Mas existe uma diferença fundamental:

> na GSNA, o produtor dos acessos também pode aprender a gerar padrões de acesso melhores.

---

# 46. Objetivo principal

O objetivo principal é:

> **Construir um sistema neural cujo custo marginal seja determinado predominantemente pelo subgrafo ativado e pelo orçamento escolhido, e não diretamente pela capacidade total armazenada.**

Idealmente:

\[
Cost(token)
\approx
Cost(microcore)
+
Cost(active\ capabilities)
+
Cost(required\ I/O)
\]

---

# 47. Objetivos mensuráveis

## Qualidade

```text
perplexity
reasoning
coding
factuality
planning
multimodal tasks
long-term memory
```

## Computação

\[
FLOPs/token
\]

## Ativação

\[
\frac{ActiveParameters}{TotalParameters}
\]

## Residência

\[
\frac{ResidentParameters}{TotalParameters}
\]

## I/O

\[
BytesFetched/token
\]

## Índice

```text
resident index bytes
lookup latency
nodes examined/query
```

## Cache

```text
hit ratio
reuse distance
evictions
```

## Roteamento

```text
entropy
load balance
route stability
route depth
```

## Treinamento

```text
resident optimizer bytes
resident gradient bytes
nodes updated/batch
I/O/update
```

---

# 48. Hipótese de escalabilidade

O experimento mais importante não é simplesmente aumentar parâmetros.

É aumentar capacidade total mantendo o custo ativo aproximadamente estável.

Exemplo:

```text
GSNA-A
100M total
32M resident

GSNA-B
200M total
33M resident

GSNA-C
400M total
35M resident

GSNA-D
800M total
37M resident
```

e observar:

```text
quality               ↑
resident bytes         ~
FLOPs/token            ~
bytes/token            ~
```

Essa seria evidência direta da hipótese estrutural.

---

# 49. Capacidade 100× maior não implica inteligência 100×

Essa distinção é obrigatória.

Se o sistema passar de:

```text
1 TB de capacidade
```

para:

```text
100 TB de capacidade
```

não segue que ele seja:

```text
100× mais inteligente
```

Capacidade adicional pode produzir:

- especialização;
- cobertura;
- conhecimento implícito;
- melhor reasoning;
- melhores ferramentas;
- redundância;
- capacidades novas.

A relação entre parâmetros totais e inteligência precisa ser medida.

Portanto, a hipótese correta é:

> **o mesmo hardware pode potencialmente endereçar um sistema muito mais pesado, e devemos investigar quanto dessa capacidade adicional se converte em qualidade útil.**

---

# 50. O computador pequeno

Considere uma máquina comum com:

```text
RAM limitada
GPU pequena ou ausente
SSD relativamente grande
```

Uma GSNA madura poderia manter:

```text
microcore
+
índice quente
+
working cache
```

na memória disponível.

O restante permanece em storage.

Conceitualmente:

```text
1 GiB working set
       │
       ▼
10 GB / 100 GB / TB de capabilities em disco
```

Isso poderia permitir acesso a sistemas muito maiores que a RAM física.

A velocidade dependeria de:

- bandwidth;
- cache hit rate;
- bytes/token;
- qualidade do prefetch;
- compute local.

O objetivo não é eliminar essas limitações.

É desacoplar **capacidade disponível** de **memória obrigatoriamente residente**.

---

# 51. O computador grande

O princípio também beneficia hardware de alto desempenho.

Considere um servidor que hoje consegue manter:

```text
X parâmetros
```

em memória rápida.

Uma GSNA não precisa usar essa memória apenas para fazer caber o modelo atual.

Ela pode usá-la como um enorme working set para um Capability Graph muito maior.

Conceitualmente:

```text
hardware atual
      │
      ▼
grande cache residente
      │
      ▼
Capability Graph
10×
100×
ou mais pesado
```

A razão não é mágica.

É simplesmente:

\[
ResidentCapacity
\neq
TotalCapacity
\]

---

# 52. Frontier-scale systems

Para laboratórios de fronteira, provedores de cloud e grandes empresas de IA, a propriedade economicamente interessante seria:

> **aumentar a capacidade lógica sem exigir crescimento proporcional de HBM e VRAM por instância.**

Se uma infraestrutura consegue atender um modelo residente de tamanho \(X\), uma arquitetura altamente esparsa e paginável poderia investigar capacidades totais:

\[
10X,\ 50X,\ 100X
\]

mantendo apenas uma fração ativa.

Isso não significa que o sistema ficaria automaticamente 10×, 50× ou 100× melhor.

Significa que o hardware passaria a poder **endereçar um espaço de capacidade muito maior**.

---

# 53. Uma nova curva econômica

Hoje, aproximadamente:

```text
mais capacidade
     ↓
mais memória rápida
     ↓
mais GPUs
     ↓
mais custo
```

A GSNA tenta criar:

```text
mais capacidade total
     ↓
mais storage
+ melhor routing
+ melhor locality
     ↓
crescimento muito menor
do working set rápido
```

Storage é significativamente mais barato e mais abundante que memória de alta velocidade.

Se a quantidade de bytes movimentados por token puder ser mantida baixa, parte do scaling pode migrar de:

```text
HBM-bound
```

para:

```text
addressability + locality + storage hierarchy
```

---

# 54. Impacto sobre acesso individual

Uma arquitetura desse tipo poderia reduzir a distância entre:

```text
modelo disponível
```

e:

```text
modelo executável localmente
```

Potenciais consequências:

- IA local mais capaz;
- maior privacidade;
- menor dependência de cloud;
- uso offline;
- modelos especializados distribuíveis;
- maior vida útil de hardware antigo;
- acesso educacional mais amplo.

---

# 55. Impacto sobre empresas

Empresas poderiam distribuir:

```text
microcore comum
+
capability packs privados
+
memory graphs locais
```

Isso permitiria sistemas especializados sem duplicar uma base inteira de modelo para cada domínio.

Possíveis usos:

- medicina;
- direito;
- engenharia;
- ciência;
- software;
- indústria;
- finanças;
- operações internas.

---

# 56. Impacto sobre cloud

Clouds poderiam separar:

```text
hot capability tier
warm capability tier
cold capability tier
```

e compartilhar capabilities imutáveis entre múltiplas sessões.

O custo de serving poderia passar a depender mais de:

```text
working set real
```

e menos de:

```text
tamanho lógico completo
```

---

# 57. Impacto sobre treinamento

Sparse Training State e Page-Major Training podem permitir treinamento de espaços de capacidade maiores que a memória agregada imediatamente disponível para optimizer state.

Isso poderia abrir pesquisa em:

- modelos muito grandes com ativação extrema;
- especialização modular;
- crescimento estrutural;
- treinamento incremental;
- capability packs independentes.

---

# 58. Impacto sobre evolução de modelos

Hoje uma nova geração frequentemente exige treinar ou distribuir um checkpoint monolítico.

Uma arquitetura modular pode permitir:

```text
base estável
+
novas capabilities
+
novas memórias
+
novos índices
```

A evolução do sistema pode tornar-se mais incremental.

---

# 59. Impacto sobre propriedade e distribuição

Capabilities poderiam, em princípio, possuir:

```text
version
identity
license
signature
provenance
compatibility
```

Isso permitiria ecossistemas modulares.

Um sistema poderia carregar:

```text
capability de linguagem
capability de código
capability científica
capability corporativa privada
```

sem necessariamente fundir tudo em um único checkpoint.

---

# 60. Impacto científico

A GSNA muda a pergunta de scaling.

Em vez de apenas:

> Quantos parâmetros conseguimos treinar?

passamos a perguntar:

> Quantos parâmetros conseguimos tornar endereçáveis mantendo pequeno o custo de cada operação?

E:

> Até onde qualidade continua melhorando quando capacidade total cresce, mas atividade e residência permanecem limitadas?

Isso é uma hipótese científica mensurável.

---

# 61. Primeiro protótipo

A primeira implementação deve ser pequena.

Exemplo:

```text
hidden dimension:     512
microcore:            5M–20M
capability nodes:     128–512
node parameters:      0.5M–2M
active nodes/step:    2–4
max graph steps:      8–16
total parameters:     50M–200M
```

Inicialmente tudo pode permanecer residente.

O primeiro objetivo é provar o routing e a aprendizagem.

---

# 62. Primeira pergunta experimental

> **Um microcore com grafo de capacidades dinamicamente roteado consegue aprender modelagem autoregressiva competitiva?**

Se não, a arquitetura precisa ser revisada antes de paging.

Nenhuma otimização de storage deve mascarar uma arquitetura neural inadequada.

---

# 63. Segunda pergunta experimental

> **Aumentar o número total de capabilities melhora qualidade sem aumentar proporcionalmente parâmetros ativos?**

Comparar:

```text
1× nodes
2× nodes
4× nodes
8× nodes
```

mantendo aproximadamente constante:

```text
active nodes
compute/token
```

---

# 64. Terceira pergunta experimental

> **A qualidade permanece quando a maior parte dos parâmetros deixa de estar residente?**

Agora introduzir:

```text
cache
pager
eviction
leases
parameter store
```

e medir:

\[
Resident
\ll
Total
\]

---

# 65. Quarta pergunta experimental

> **O modelo aprende localidade?**

Adicionar custos de:

```text
cache miss
bytes fetched
paging wait
prefetch waste
```

e observar se trajetórias mudam.

---

# 66. Quinta pergunta experimental

> **O mesmo checkpoint funciona sob diferentes budgets?**

Testar:

```text
16 MiB
32 MiB
64 MiB
128 MiB
```

e medir degradação.

Uma arquitetura robusta deve degradar gradualmente.

---

# 67. Sexta pergunta experimental

> **O índice também pode crescer sem se tornar um novo core gigante?**

Medir:

```text
total capabilities
total index size
resident index size
lookup latency
lookup accuracy
```

Esse teste é crítico.

Sem ele, o problema de memória apenas migra dos pesos para o diretório.

---

# 68. Sétima pergunta experimental

> **Training State pode ser realmente paginado?**

Construir um treinamento no qual:

\[
weights + optimizer
>
RAM
\]

mas:

\[
resident\ training\ state
<
RAM
\]

e verificar convergência e throughput.

---

# 69. Baselines obrigatórios

Comparar pelo menos:

```text
Dense Transformer
MoE Transformer
SSM / recurrent moderno
GSNA
```

com orçamento de treinamento comparável.

Não basta comparar parâmetros totais.

Medir:

```text
training FLOPs
inference FLOPs
active params/token
resident bytes
bytes/token
latency
quality
training-state residency
```

---

# 70. Critérios de sucesso

## Critério 1

Qualidade competitiva em pequena escala.

## Critério 2

Routing esparso estável.

## Critério 3

Capacidade total cresce mais rápido que parâmetros ativos.

## Critério 4

Resident bytes crescem muito mais lentamente que total parameters.

## Critério 5

Bytes/token permanecem controlados.

## Critério 6

O mesmo sistema opera sob múltiplos budgets.

## Critério 7

Índice total cresce sem exigir residência total.

## Critério 8

Training State pode ser materializado sob demanda.

---

# 71. Uma prova estrutural forte

Um resultado particularmente convincente seria:

```text
Total capacity       1× → 8×
Quality              melhora
Active parameters    ~constante
Resident bytes       ~constante
FLOPs/token          ~constante
Bytes/token          ~constante
```

Isso demonstraria que a arquitetura está conseguindo converter capacidade externa em utilidade sem carregar proporcionalmente o custo físico.

---

# 72. Métrica de eficiência de scaling

Uma métrica conceitual:

\[
ScalingEfficiency
=
\frac{\Delta Quality}
{\Delta ResidentBytes
+
\Delta FLOPs
+
\Delta IO}
\]

O objetivo é maximizar ganho de capacidade útil por unidade adicional de custo ativo.

---

# 73. Riscos científicos

## Routing collapse

O sistema usa sempre os mesmos nodes.

## Fragmentação

Existem muitos módulos pouco úteis.

## I/O domination

A arquitetura é esparsa, mas storage domina a latência.

## Router overhead

A busca custa mais que o compute economizado.

## Index explosion

O índice cresce até se tornar outro modelo residente.

## Training instability

Roteamento discreto dificulta otimização.

## Catastrophic routing changes

Pequenos updates alteram trajetórias inteiras.

## Prefetch waste

Muitas páginas são carregadas e não utilizadas.

## Microcore bottleneck

O núcleo pequeno limita inteligência global.

## Resource overfitting

A arquitetura aprende um hardware específico.

## Structural churn

Split/merge excessivo impede estabilidade.

## Sparse update starvation

Capabilities raras recebem pouco treinamento.

---

# 74. O microcore não deve ser pequeno por dogma

Existe um tamanho mínimo necessário para:

- representar estado;
- integrar resultados;
- produzir queries;
- controlar routing;
- manter linguagem basal;
- operar memória.

Portanto:

> **o melhor microcore é o menor que preserva coordenação suficiente, não o menor possível.**

---

# 75. O storage não é RAM

A arquitetura não elimina física.

Se cada token exigir:

```text
4 GB
```

de leitura, um SSD será gargalo.

Por isso, o sucesso depende de:

```text
bytes/token baixos
reuso alto
prefetch correto
rotas previsíveis
cache eficiente
nodes de granularidade adequada
```

Paging só funciona quando existe localidade suficiente.

---

# 76. Granularidade dos nodes

Nodes pequenos:

```text
+ menos bytes por miss
+ maior especialização
- mais routing
- mais metadata
- mais fragmentação
```

Nodes grandes:

```text
+ melhor compute density
+ menos routing
- mais bytes por miss
- maior desperdício
```

A granularidade ideal precisa ser descoberta experimentalmente.

---

# 77. Integridade

Parâmetros externos precisam ser tratados como objetos verificáveis.

Cada capability pode possuir:

```text
capability_id
version
size
checksum
format
dependencies
```

Falha de leitura ou corrupção não deve produzir compute parcial silencioso.

---

# 78. Segurança de atualização

Capabilities modulares podem evoluir independentemente.

O runtime deve ser capaz de:

```text
load candidate
validate
activate
rollback
```

sem comprometer o microcore ou o restante do sistema.

---

# 79. Possíveis melhorias futuras

Entre as extensões de longo prazo estão:

- hierarchical predictive routing;
- learned graph packing;
- multiresolution weights;
- capability distillation;
- adaptive precision;
- dynamic graph growth;
- learned cache policy;
- remote capability execution;
- distributed capability fabrics;
- federated capability updates;
- hardware-specific execution planners;
- neural compression of routing indices;
- learned address structures;
- speculative capability execution;
- asynchronous memory consolidation.

---

# 80. Capability Distillation

Capabilities muito utilizadas podem ser parcialmente destiladas para regiões mais próximas do microcore.

Exemplo:

```text
capability externa muito quente
       ↓
compact summary/residual
       ↓
hot local capability
```

O sistema pode adaptar sua estrutura física ao uso real.

---

# 81. Dynamic Precision

Uma mesma capability pode existir em múltiplas precisões.

O runtime pode decidir:

```text
Q2
Q4
Q6
FP16
```

de acordo com:

- budget;
- importância;
- sensitividade;
- hardware;
- profundidade desejada.

Isso adiciona outro eixo de elasticidade.

---

# 82. Distribuição em larga escala

Em clusters, o Capability Fabric pode ser distribuído.

```text
microcore
   │
   ├── local VRAM
   ├── local RAM
   ├── node-local NVMe
   ├── rack-local capability store
   └── remote capability store
```

Routing e residency tornam-se um problema conjunto de rede, memória e compute.

---

# 83. A arquitetura como sistema operacional neural

Uma analogia útil é:

```text
microcore        → kernel neural
Capability IDs   → virtual addresses
Capability Graph → addressable program space
resident cache   → working set
pager            → memory manager
router           → scheduler/planner
Memory Graph     → persistent structured state
```

Não é uma equivalência literal.

Mas expressa a mudança conceitual:

> o modelo deixa de ser apenas um tensor estático carregado em memória e passa a ser um espaço de capacidades administrado dinamicamente.

---

# 84. Potencial de democratização

Se capacidade total puder ser separada do working set obrigatório, o piso de hardware para acessar sistemas avançados pode cair.

Isso pode permitir:

```text
hardware pequeno
→ menor velocidade
→ mesmo espaço lógico de capacidades
```

enquanto:

```text
hardware grande
→ mais cache
→ mais paralelismo
→ maior profundidade
→ maior velocidade
```

A diferença entre máquinas passa a afetar principalmente **quanto do sistema pode permanecer próximo do compute e quão rápido ele pode navegar**, não necessariamente quais capacidades existem no storage.

---

# 85. Potencial de superescala

No outro extremo, uma infraestrutura de 2026 capaz de hospedar um grande modelo residente poderia usar o mesmo hardware como working set para uma arquitetura logicamente muito maior.

A hipótese é:

```text
mesmo hardware rápido
      │
      ▼
não apenas modelo X
      │
      ▼
mas Capability Graph
muitas vezes maior que X
```

O objetivo científico seria medir se um aumento de 10×, 100× ou mais na capacidade total gera ganhos úteis mantendo:

```text
active compute
resident bytes
bandwidth/token
```

sob controle.

---

# 86. Potencial impacto sobre a fronteira da IA

Se a arquitetura funcionar em escala, os limites de frontier models podem mudar.

Hoje um limite importante é:

> quanto do modelo pode ser treinado e servido economicamente próximo do compute?

Em GSNA, a pergunta passa a ser:

> quanto de capacidade conseguimos endereçar eficientemente por meio de um working set limitado?

Isso poderia incentivar modelos com:

- muito mais especializações;
- grandes bibliotecas neurais;
- capacidades de domínio profundas;
- crescimento incremental;
- memória persistente extensa;
- compute adaptativo.

---

# 87. Um novo tipo de scaling law

Scaling tradicional observa relações entre:

```text
parameters
data
compute
loss
```

GSNA adiciona variáveis:

```text
total capabilities
active capabilities
resident capabilities
bytes/token
routing depth
cache reuse
index size
storage latency
```

Uma questão científica nova seria:

\[
Quality
=
f(
TotalCapacity,
ActiveCapacity,
ResidentCapacity,
IO,
Routing,
Data,
Compute
)
\]

Descobrir essa função pode ser tão importante quanto construir a arquitetura.

---

# 88. Meta de longo prazo

A meta não é simplesmente:

> rodar um modelo de 2 TB em 1 GB.

Uma formulação mais rigorosa é:

> **construir uma arquitetura na qual um working set pequeno possa navegar e utilizar corretamente um espaço de capacidade muitas ordens de magnitude maior, pagando principalmente pelo que é realmente ativado.**

Isso preserva os limites físicos.

E transforma a maneira como capacidade é organizada.

---

# 89. Programa experimental

A pesquisa deve avançar incrementalmente.

## Fase 0 — Formalização

Definir:

- microcore;
- state representation;
- Capability Node contract;
- Capability Address Space;
- hierarchical router;
- loss;
- budgets;
- métricas.

## Fase 1 — Graph LM mínimo

Tudo residente.

Provar linguagem.

## Fase 2 — Sparse Capability Graph

Aumentar capacidade mantendo ativação pequena.

## Fase 3 — Adaptive Compute

Treinar stop/continue.

## Fase 4 — Hierarchical Capability Addressing

Provar navegação sem índice monolítico.

## Fase 5 — Simulated Budget Training

Variar RAM, bandwidth e compute.

## Fase 6 — Physical Paging

Mover capabilities para storage.

## Fase 7 — Predictive Prefetch

Treinar previsão de futuras capacidades.

## Fase 8 — Hardware-Aware Routing

Incorporar custos físicos reais.

## Fase 9 — Sparse Training State

Paginar optimizer state e gradientes.

## Fase 10 — Page-Major Training

Reordenar execução de treino por módulos.

## Fase 11 — Multiresolution Capabilities

Adicionar níveis de precisão/fidelidade.

## Fase 12 — Memory Graph

Adicionar conhecimento persistente explícito.

## Fase 13 — Structural Growth

Investigar spawn/split/merge/prune.

## Fase 14 — Large-Scale Fabric

Validar distribuição entre múltiplos tiers e máquinas.

---

# 90. Primeiro milestone

O primeiro milestone deve ser pequeno e brutalmente verificável:

> **Um GSNA residente consegue competir com um baseline de tamanho semelhante enquanto ativa significativamente menos parâmetros por passo.**

Sem isso, não há motivo para implementar paging.

---

# 91. Segundo milestone

> **Aumentar o Capability Graph melhora qualidade sem crescimento proporcional de active parameters.**

Esse resultado prova sparsity útil.

---

# 92. Terceiro milestone

> **Mover a maioria das capabilities para storage preserva qualidade e mantém I/O controlável.**

Esse resultado prova residency sparsity.

---

# 93. Quarto milestone

> **Um checkpoint treinado sob Resource Dropout funciona em múltiplas classes de hardware.**

Esse resultado prova elasticidade.

---

# 94. Quinto milestone

> **O total training state excede a RAM disponível sem exigir sua residência simultânea.**

Esse resultado prova a viabilidade do caminho de treinamento escalável.

---

# 95. A visão

A GSNA propõe que inteligência neural deixe de ser vista apenas como:

```text
uma enorme função
```

e passe a ser organizada como:

```text
microcore
+
estado
+
espaço de endereçamento
+
grafo de capacidades
+
memória
+
roteamento
+
runtime
+
budget físico
```

A capacidade total pode ser enorme.

A atividade instantânea permanece pequena.

O índice total pode crescer.

A parte residente do índice permanece seletiva.

Conhecimento pode crescer.

O contexto permanece limitado.

Uma tarefa simples utiliza poucos recursos.

Uma tarefa difícil pode percorrer regiões maiores.

Um computador pequeno pode acessar um sistema muito maior do que sua RAM.

Um computador grande pode usar sua memória rápida como working set de um sistema ainda maior.

---

# 96. Hipótese final

A hipótese final da GSNA é:

> **O limite útil de uma inteligência neural não precisa ser determinado pela quantidade de parâmetros que podem permanecer simultaneamente em RAM ou VRAM.**

Se parâmetros, capacidades, memória e índices puderem ser endereçados hierarquicamente; se apenas um subgrafo pequeno precisar ser materializado; se o routing aprender localidade e previsibilidade; e se o treinamento incorporar explicitamente os custos físicos da máquina, então:

\[
TotalCapacity
\]

pode crescer muito mais rapidamente que:

\[
ResidentCapacity
\]

e potencialmente também mais rapidamente que:

\[
Compute/token
\]

e:

\[
Bytes/token
\]

A consequência possível é uma nova classe de sistemas neurais:

```text
pequenos no estado ativo
grandes no espaço de capacidade
adaptáveis ao hardware
modulares
pagináveis
expansíveis
```

---

# 97. Princípio fundador

A GSNA pode ser resumida em uma única afirmação:

> **O tamanho da inteligência disponível não deveria ser determinado pelo tamanho do seu working set físico.**

Ou, formalmente:

\[
\boxed{
SystemCapacity
\neq
ActiveComputation
\neq
ResidentMemory
}
\]

O desafio científico é descobrir até onde essa separação pode ser levada sem perder qualidade, velocidade ou capacidade de aprendizado.

Se a resposta for suficientemente longe, a relação entre IA e hardware pode mudar de forma fundamental.
