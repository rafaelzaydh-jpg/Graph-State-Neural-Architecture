**GRAPH-STATE  
NEURAL ARCHITECTURE**

Da otimização de modelos existentes a uma arquitetura neural construída para capacidade endereçável

**WHITE PAPER** Versão 0.3 — revisão RNC-128MB  
**STATUS** Especificação arquitetural de convergência, decisões abertas e programa experimental  
**BASE** Eyle 2.7.5 Rev4.2.3 • MiniMaxBrain 0.4.1 • GSNA v0.1 • GSNA v0.2

**Tese central**

**Não construir um modelo gigante e depois otimizar para que ele caiba. Construir uma arquitetura em que inteligência residente, capacidade endereçável e computação ativa possam escalar separadamente.**

**Baseline desta revisão: RNCWeightBytes = 128 MB. É um ponto experimental, não um teto arquitetural.  
**

# Resumo

A Graph-State Neural Architecture (GSNA) nasce de uma sequência de tentativas de desacoplar capacidade lógica de custo físico.

O **Eyle** mostrou, em uma aplicação funcional, que memória persistente pode crescer sem que toda a memória seja automaticamente materializada no contexto. O **MiniMaxBrain (MMB)** estendeu o mesmo princípio aos parâmetros de modelos Mixture-of-Experts: um modelo pode possuir experts logicamente disponíveis sem manter todos os seus pesos simultaneamente residentes em RAM, desde que esses parâmetros sejam endereçáveis, verificáveis, pagináveis, reutilizáveis e protegidos durante a computação.

Essas aplicações também expuseram o limite da otimização externa.

Depois de cache, I/O, paralelismo, leases, integridade, batching, prefetch e instrumentação, permanece uma restrição estrutural: o modelo original não foi treinado para viver sob esse regime físico. O runtime consegue materializar de forma mais eficiente aquilo que o modelo pede, mas não controla quais capacidades são pedidas, em que ordem, com que localidade ou com qual previsibilidade.

A GSNA ataca esse problema na origem:

**em vez de construir um modelo gigante e depois otimizar sua execução para caber no hardware, construir desde o início uma rede cuja capacidade, memória, computação ativa e residência física sejam recursos endereçáveis.**

A revisão v0.3 também corrige uma distorção introduzida durante a evolução da v0.2: o antigo **Resident Neural Core** não deve ser entendido como um hub obrigatório nem como um núcleo que precisa permanecer artificialmente pequeno.

A terminologia canônica passa a ser **Resident Neural Core (RNC)**.

O RNC é a parte neural que permanece residente e fornece inteligência basal, estado temporal, entrada/saída, suporte ao roteamento, integração e fallback. Seu tamanho é uma **dimensão independente de escala**. A GSNA não exige que o RNC permaneça constante quando o Capability Graph cresce e tampouco exige que cresça proporcionalmente a ele.

A relação arquitetural desejada é:

\|C\| ⟂arch \|Gₚ\|

onde:

- \(C\) representa o **Resident Neural Core**;

- (G_P) representa o **Parameter/Capability Graph**;

- ⟂arch significa desacoplamento arquitetural, e não independência estatística: não existe uma regra estrutural que obrigue o tamanho de C a ser proporcional ao tamanho de Gₚ.

O primeiro baseline desta versão será um **RNC com orçamento de 128 MB de pesos residentes**. Esse valor é um ponto experimental, não um limite da arquitetura.

O RNC contém um **Fast State Fabric**, uma sub-rede residente para transportar e atualizar estado causal de baixo volume. O estado neural pode seguir diretamente de uma capability para outra; não há requisito de retornar ao RNC entre cada transformação.

O **Capability Graph** contém transformações neurais endereçáveis e potencialmente não residentes. O **Memory Graph**, herdado conceitualmente do Eyle, contém conhecimento explícito, mutável, verificável e revisável. O **runtime GSNA**, apoiado nos mecanismos já demonstrados pelo MMB, é responsável por materialização física, cache, leases, integridade, telemetria e paging.

O objetivo da v0.3 não é declarar a arquitetura resolvida. É separar com precisão:

1.  o que Eyle e MMB já responderam;

2.  o que a GSNA realmente precisa inventar;

3.  quais decisões ainda bloqueiam um Graph Language Model funcional;

4.  quais testes podem falsificar ou fortalecer a hipótese;

5.  como medir separadamente **Core Scaling** e **Capability Scaling**.

A pergunta científica central passa a ser:

**quanto de inteligência vale a pena manter residente, quanto pode existir como capacidade endereçável e como essas duas escalas interagem sob um orçamento físico real?**

# 1. Tese do projeto

A hipótese da GSNA não é simplesmente que Transformers sejam ineficientes.

Transformers são extremamente eficientes quando avaliados dentro da premissa para a qual foram desenhados: uma topologia relativamente fixa, altamente vetorizável, com grande quantidade de parâmetros e estado próximos ao dispositivo de compute.

A hipótese da GSNA é outra:

**Para uma inteligência cuja capacidade total deva crescer muito além do working set físico disponível, uma topologia obrigatoriamente fixa pode ser a abstração errada.**

Em uma arquitetura tradicional, uma sequência atravessa uma pilha conhecida de operações. Mesmo quando existem experts esparsos, a organização macro permanece amplamente determinada pelas layers.

Na GSNA, a computação passa a ser descrita por:

\[ \]

A topologia executada deixa de ser completamente conhecida antes de observar o estado.

O objetivo não é eliminar toda estrutura fixa. É limitar a estrutura fixa ao que precisa ser sempre residente e sempre confiável, permitindo que a maior parte da capacidade exista como um espaço endereçável.

A hipótese pode ser resumida por seis desacoplamentos:

**Memory size ≠ Context size.**

**Model size ≠ Resident model size.**

**System capacity ≠ Active computation.**

**Training state size ≠ Resident training state.**

**Capability size ≠ Bandwidth per token.**

**Graph size ≠ Path length.**

A sexta relação é explicitada nesta versão: aumentar o número total de capacidades não deve obrigar cada token a percorrer proporcionalmente mais nós.

# 2. Genealogia: como a GSNA surgiu

## 2.1 Eyle — memória total não é contexto ativo

O Eyle foi a primeira evidência operacional da ideia de endereçamento seletivo.

Seu Memory Graph mantém estado persistente com propriedades como:

- identidade;

- scope;

- domain;

- context identity;

- natureza epistemológica;

- confiança;

- volatilidade;

- temporalidade;

- relações;

- proveniência;

- revisão;

- freshness;

- histórico de mudanças.

O ponto arquitetural mais importante não é a existência de uma base de memória.

É a regra:

    graph growth
        !=
    automatic prompt growth

A implementação separa persistência de materialização. O contexto cognitivo é projetado sob budgets explícitos. Nós não materializados permanecem alcançáveis por identidade, recall ou ativação explícita.

O Eyle também introduz uma separação útil entre semântica e mecânica. O Capability Registry expõe capacidades canônicas, seus contratos, efeitos, disponibilidade e executores, mas não se torna um segundo planejador semântico.

Dessa experiência a GSNA herda três princípios:

1.  **recursos podem ser numerosos sem estar todos presentes no estado ativo;**

2.  **endereçamento e materialização devem ser operações distintas;**

3.  **o runtime deve possuir autoridade mecânica sem usurpar a decisão semântica.**

## 2.2 MiniMaxBrain — modelo lógico não é modelo residente

O MiniMaxBrain aplicou o mesmo raciocínio aos pesos neurais.

No caminho nativo, routed expert weights podem existir logicamente no grafo do modelo enquanto o payload completo permanece fora da memória física. O router neural continua sendo a autoridade sobre quais expert IDs são usados. O pager apenas materializa os bytes correspondentes.

O fluxo funcional é, de forma simplificada:

    llama.cpp / router neural
            |
            v
    expert IDs
            |
            v
    MMBPager
            |
            +-- cache hit -> lease/pin
            |
            +-- cache miss
                    |
                    +-- reserve
                    +-- read
                    +-- verify integrity
                    +-- atomic commit
                    +-- lease
            |
            v
    GGML compute
            |
            v
    release lease

O MMB estabeleceu mecanismos que não precisam ser reinventados pela GSNA:

- parameter store;

- metadata-only logical placeholders;

- pager;

- cache;

- leases;

- eviction safety;

- atomic batch commit;

- integrity verification;

- fail-closed behavior;

- native telemetry;

- separation between router authority and materialization;

- measurement of cache hits, misses, bytes fetched, I/O wait and critical path.

O MiniMaxBrain também mostrou o limite da otimização de uma arquitetura que não foi treinada para paging.

No benchmark A2 incluído no pacote, a execução warm do candidato aumentou a mediana de aproximadamente **0,49 tok/s para 1,06 tok/s** em relação ao baseline, mantendo o mesmo output medido. Porém, o paging ainda representava aproximadamente **68%** do tempo de decode medido no candidato. A otimização melhorou muito a execução, mas não mudou o padrão de acesso produzido pelo modelo.

Esse resultado é uma das motivações mais fortes da GSNA:

**o runtime pode tornar um padrão de acesso menos ruim; somente o treinamento pode ensinar o modelo a produzir um padrão de acesso diferente.**

## 2.3 GSNA v0.1 — computação como caminho dinâmico

O whitepaper v0.1 generalizou os dois princípios anteriores.

Em vez de:

    Layer 1
      |
    Layer 2
      |
    Layer 3
      |
    ...
      |
    Layer N

a proposta passou a ser:

    Node A
     /   \
    B     C
     \   /
       D
       |
      E/F

Cada operação percorre apenas uma região do grafo.

Formalmente:

\[ h\_{k+1}=f\_{v_k}(h_k;\_{v_k}) \]

e:

\[ V\_{k+1}^{active}=R(h_k,S_k,G) \]

A contribuição central da v0.1 foi, portanto:

**a própria organização da computação pode ser endereçável.**

## 2.4 GSNA v0.2 — treinamento sob orçamento físico e uma correção necessária

A versão 0.2 tornou explícitas quatro ideias importantes:

- Neural Microcore;

- Capability Fabric;

- Budget-Constrained Neural Training;

- Sparse Training State.

Ela também elevou métricas físicas, especialmente BytesFetched/token, ao centro do problema e propôs que o modelo aprenda localidade, prefetch, residency e uso de compute.

Entretanto, a depuração conceitual da v0.2 introduziu duas conclusões que não pertenciam à essência da proposta original:

1.  o RNC passou a poder ser interpretado como um **hub obrigatório** pelo qual toda transformação deveria retornar;

2.  “manter o RNC pequeno” passou a aparecer quase como um objetivo arquitetural.

A v0.3 corrige explicitamente as duas interpretações.

O RNC é residente, mas não é um hub obrigatório.

E o RNC pode crescer quando isso melhora qualidade, roteamento, integração ou capacidade basal. O que a GSNA exige é **desacoplamento**, não miniaturização.

Em outras palavras:

\[ \]

e também:

\[ \]

A escala correta do RNC deve ser descoberta experimentalmente.

# 3. O que já está respondido

A GSNA não deve reabrir problemas que já possuem resposta operacional nas aplicações predecessoras.

## 3.1 Memória persistente e projeção seletiva

**Respondido principalmente pelo Eyle.**

A GSNA pode reutilizar os seguintes conceitos:

- identidade persistente;

- relações explícitas;

- revisão;

- proveniência;

- temporalidade;

- freshness;

- alcance;

- materialização sob budget;

- separação entre memória persistida e memória ativa.

A pesquisa GSNA não precisa provar novamente que um grafo de memória pode crescer sem entrar integralmente no contexto.

O problema novo é definir **como estado neural consulta e recebe resultados desse grafo sem converter todo recall em texto de prompt**.

## 3.2 Registry e contratos de capacidades

**Respondido parcialmente pelo Eyle.**

A noção de capacidade endereçável com identidade canônica, input contract, output contract, availability, effect e provider ownership já possui implementação concreta.

A GSNA deve adaptar esse princípio para nós neurais.

Um Capability Node precisa possuir, no mínimo:

    node_id
    node_type
    state_contract
    parameter_contract
    routing_key
    dependencies
    residency_state
    version
    integrity_metadata
    physical_tier

O registry continua sendo infraestrutura mecânica.

A semântica de qual capacidade utilizar pertence ao roteamento aprendido.

## 3.3 Paging e residência de parâmetros

**Respondido em grande parte pelo MiniMaxBrain.**

A GSNA deve herdar, salvo evidência contrária:

- parameter store;

- page metadata;

- checksums;

- versioning;

- cache;

- LRU ou política substituível;

- leases;

- pin durante compute;

- atomic load/commit;

- explicit failure;

- no silent fallback;

- métricas de paging;

- bundle verification.

O problema novo não é como ler uma página do SSD.

O problema novo é:

**como treinar trajetórias que necessitem de poucas páginas, reutilizem páginas carregadas e sejam previsíveis o suficiente para prefetch.**

## 3.4 Autoridade do router

**Respondido como invariante pelo MiniMaxBrain.**

No MMB, o pager não escolhe experts. O router neural é a autoridade semântica e o runtime é a autoridade física.

A mesma separação deve valer para a GSNA:

    Neural policy
        |
        | escolhe intenção computacional
        v
    node IDs / route distribution
        |
        v
    Runtime
        |
        | decide materialização física válida
        v
    resident pages / execution

O runtime pode recusar uma rota impossível sob hard constraints, mas não deve silenciosamente substituir uma capacidade por outra semanticamente diferente.

## 3.5 Integridade, telemetria e verdade física

**Respondido principalmente pelo MiniMaxBrain.**

A GSNA deve nascer instrumentada.

Métricas ausentes não podem ser fabricadas como zero. Service time não deve ser confundido com wall time. Comparações precisam usar janelas temporais coerentes. Toda otimização deve ser selecionada a partir do caminho crítico real.

Esse princípio é parte da arquitetura de pesquisa, não somente engenharia de benchmark.

# 4. Resident Neural Core (RNC)

A v0.3 substitui o termo **Resident Neural Core** por **Resident Neural Core (RNC)**.

A mudança não é cosmética.

“Microcore” carregava uma conclusão prematura: a de que o núcleo deveria ser o menor possível. A propriedade realmente necessária é outra:

**o núcleo deve ser residente, confiável e útil; seu tamanho deve ser escolhido pela melhor relação entre qualidade e custo, não por uma obrigação de permanecer pequeno.**

A representação de alto nível passa a ser:

\[ System_t=(C,F,G_P,G_M,H_t,R_t,B_t) \]

onde:

- (C): **Resident Neural Core (RNC)** — a capacidade neural sempre residente;

- (F): **Fast State Fabric** — sub-rede residente, logicamente pertencente ao RNC, responsável por transporte e atualização rápida de estado;

- (G_P): **Parameter/Capability Graph** — grafo de capacidades neurais endereçáveis;

- (G_M): **Memory Graph** — memória explícita, persistente, revisável e endereçável;

- (H_t): **Fast Temporal State** — estado causal necessário para processar o token (t);

- (R_t): estado de roteamento, residência e cache relevante para a decisão corrente;

- (B_t): **orçamento físico** disponível no instante (t), como RAM/VRAM disponível, bandwidth, compute e limites de I/O.

O Fast State Fabric é mostrado separadamente porque possui um contrato específico, mas fisicamente pode fazer parte do mesmo checkpoint residente do RNC:

\[ F C \]

A arquitetura desejada é:

                             GSNA
                              |
                    Resident Neural Core
              +---------------+----------------+
              |               |                |
         basal neural     Fast State       routing /
          capacity          Fabric         integration
              |               |
              |               +-----------------------------+
              |                                             |
              v                                             v
          State Packet --------------------------> Capability Graph
                                                      /   |   \
                                                     /    |    \
                                                    v     v     v
                                                  Node -> Node -> Node
                                                    \           /
                                                     \         /
                                                      +-------+
                                                          |
                                                          v
                                                    state commit
                                                          |
                                              +-----------+-----------+
                                              |                       |
                                              v                       v
                                          LM Head               Memory Gateway
                                                                      |
                                                                      v
                                                                 Memory Graph

Não existe a regra:

    Capability -> RNC -> Capability -> RNC -> Capability

Uma capability pode encaminhar estado diretamente para outra capability quando a topologia e a política de routing permitirem.

O RNC fornece substrato residente, estado e coordenação. Ele não precisa ser o gargalo de toda comunicação.

## 4.1 Baseline RNC-128MB

A primeira implementação será denominada **RNC-128MB**.

Nesta versão, “128 MB” significa:

\[ B\_{RNC,w}=128  \]

onde (B\_{RNC,w}) é o **budget de bytes dedicado aos pesos residentes do RNC**.

Esse número não significa “128 milhões de parâmetros”.

A quantidade de parâmetros que cabe nesse budget depende da representação numérica. Ignorando pequenos overheads de metadata:

| **Representação dos pesos** | **Bytes aproximados por parâmetro** | **Parâmetros que cabem em ~128 MB** |
|-----------------------------|-------------------------------------|-------------------------------------|
| FP32                        | 4                                   | ~32M                                |
| BF16 / FP16                 | 2                                   | ~64M                                |
| INT8                        | 1                                   | ~128M                               |
| 4-bit                       | 0,5                                 | ~256M                               |

Esses números são apenas equivalências de armazenamento. Eles **não** dizem que todas as precisões possuem a mesma qualidade, throughput ou custo de treinamento.

Para evitar métricas enganosas, a GSNA deve reportar separadamente:

\[ RNCWeightBytes \]

bytes dos pesos do RNC;

\[ RNCStateBytes \]

bytes do estado temporal e de roteamento;

\[ RNCActivationBytes \]

ativações temporárias relevantes;

\[ RNCRuntimeBytes \]

buffers e estruturas do runtime residentes por causa do RNC.

A memória residente total atribuível ao RNC é, portanto:

\[ RNCTotalResidentBytes = RNCWeightBytes + RNCStateBytes + RNCActivationBytes + RNCRuntimeBytes \]

O baseline “128 MB” fixa inicialmente apenas RNCWeightBytes.

Além disso, benchmarks devem registrar se “MB” é decimal ou se o runtime reporta “MiB” binário, para evitar comparar números diferentes sob o mesmo rótulo.

## 4.2 O RNC de 128 MB não é um teto

A escolha de 128 MB é uma âncora experimental.

Ela não estabelece:

\|C\| ≤ 128 MB

como princípio da GSNA.

A hipótese é que um RNC maior provavelmente pode melhorar qualidade. Isso não é um problema; é uma dimensão de scaling que precisa ser medida.

A pergunta correta é:

**qual aumento de qualidade recebemos para cada byte adicional mantido permanentemente residente?**

## 4.3 Core Scaling e Capability Scaling são eixos diferentes

A GSNA deve medir dois tipos de escala.

### Core Scaling

Aumentar (C), o RNC:

    64 MB
      |
    128 MB
      |
    256 MB
      |
    512 MB
      |
    1 GB

tende a aumentar capacidade basal, qualidade de estado, routing e integração, mas cobra memória e compute residentes.

### Capability Scaling

Aumentar (G_P), o Capability Graph:

    1x
     |
    2x
     |
    4x
     |
    8x
     |
    16x

aumenta capacidade potencial sem exigir que toda essa capacidade permaneça residente ou ativa.

Os eixos são arquiteturalmente desacoplados:

\|C\| ⟂arch \|Gₚ\|

O símbolo ⟂arch significa:

**o design não impõe proporcionalidade obrigatória entre os dois tamanhos.**

Ele não significa independência estatística.

Na prática, pode existir uma interação:

G_usable = f(C)

onde G_usable é a parte do Capability Graph que o RNC consegue navegar e utilizar com qualidade suficiente.

Essa função precisa ser descoberta experimentalmente.

## 4.4 Core sizing como problema de otimização

O tamanho do RNC deve ser escolhido como função de requisitos de implantação:

\[ C=C(B,Q,L) \]

onde:

- \(B\) = **orçamento físico**: RAM/VRAM, compute disponível, bandwidth, energia ou limites de I/O;

- \(Q\) = **qualidade desejada**: nível de loss/perplexidade, capacidade de reasoning, coding, factuality ou outra meta;

- \(L\) = **restrição de latência**: tempo máximo aceitável para prefill, decode, resposta ou uma operação completa.

A equação não define ainda uma função fechada. Ela expressa o problema:

para um hardware e uma meta de qualidade/latência, existe um tamanho de RNC mais vantajoso do que outros.

Em uma máquina de 8 GB, o RNC-128MB deve deixar amplo espaço para sistema operacional, Fast State, cache de capabilities, buffers e outros componentes. Em máquinas maiores, um RNC maior pode ser racional.

A GSNA deve adaptar-se ao hardware; não deve sacrificar sua arquitetura para perseguir o menor footprint imaginável.

## 4.5 Funções mínimas do RNC

O RNC deve conter somente funções cuja residência frequente tenha valor suficiente para justificar seu custo.

Candidatas:

- token embedding e representação de entrada;

- capacidade linguística basal;

- Fast State Fabric;

- estado temporal causal;

- suporte a routing local e global;

- integração de resultados;

- fallback quando uma capability externa não está disponível;

- stop/continue;

- prefetch hints;

- LM head ou projeção compartilhada, se isso for vantajoso.

A lista não significa que cada função precise ser implementada como um módulo isolado.

O objetivo é que o RNC continue sendo uma **rede neural útil por si mesma**, enquanto o Capability Graph adiciona capacidade especializada e escalável.

## 4.6 RNC como fallback funcional

Uma propriedade desejável é:

Output(C) permanece válido quando Gₚ_available é reduzido

Isso não exige que o RNC iguale a qualidade do sistema completo.

Exige degradação graciosa:

    RNC only
       -> linguagem basal

    RNC + small resident graph
       -> melhor qualidade

    RNC + large capability graph
       -> capacidade ampliada

Essa propriedade reduz acoplamento operacional e permite testar separadamente a contribuição real do Capability Graph.

# 5. State Packet: o que realmente circula

Um token ID não deve ser imaginado literalmente viajando pelo grafo.

O que circula é uma representação neural associada ao token e ao estado causal disponível.

A v0.3 denomina essa unidade lógica de **State Packet**.

Uma forma abstrata:

\[ P\_{t,k}=(h\_{t,k},q\_{t,k},b\_{t,k},\_{t,k}) \]

onde:

- (h\_{t,k}): estado neural;

- (q\_{t,k}): estado mínimo de rota;

- (b\_{t,k}): budget restante;

- (\_{t,k}): referências físicas/lógicas necessárias ao runtime.

Isso não significa que todos esses campos precisem existir como um objeto material em cada kernel. É um contrato lógico.

A propriedade central é que a carga de comunicação entre capabilities seja pequena em comparação com a carga dos parâmetros.

Se um estado tiver dimensão 1024 em BF16, o vetor principal possui aproximadamente 2 KiB. Uma página neural pode possuir dezenas de MiB.

Essa assimetria é proposital:

**mover estado deve ser barato; mover capacidade deve ser raro e reutilizável.**

# 6. Fast State Fabric

A v0.3 introduz explicitamente o **Fast State Fabric (FSF)**.

O FSF é a sub-rede residente responsável por comunicação neural rápida e causal. Ele pertence logicamente ao RNC:

\[ F C \]

mas é separado conceitualmente porque sua função é diferente da capacidade basal do núcleo.

O FSF não existe para armazenar a maior parte da inteligência.

Ele existe para transportar e atualizar um estado compacto o suficiente para que:

- o token atual receba informação causal do passado;

- capability nodes recebam um contexto neural útil;

- resultados de capabilities sejam integrados ao fluxo temporal;

- um node possa encaminhar estado diretamente a outro;

- o roteamento possa reagir ao estado sem materializar todo o histórico;

- a maior parte do Capability Graph permaneça fora do working set.

Conceitualmente:

    token embedding
          |
          v
    Fast State Fabric <--------------------------------------+
          |                                                  |
          v                                                  |
    State Packet                                             |
          |                                                  |
          +--> Capability A --> Capability B --> Node C -----+
          |                           |
          |                           +--> Node D
          |
          +--> Memory Gateway
          |
          v
    Integration / Commit
          |
          v
    Fast Temporal State
          |
          v
    LM Head

A linha Capability A -\> Capability B é importante: o estado não precisa retornar a um hub central entre cada transição.

## 6.1 Requisito temporal

Para modelagem autoregressiva:

\[ P(x\_{t+1}\|x\_{t}) \]

o FSF precisa garantir causalidade estrita.

Um ciclo abstrato é:

\[ u\_{t,0}=F\_{in}(e_t,H\_{t-1}) \]

\[ u\_{t,k+1}=u\_{t,k}+*{v_k}(u*{t,k}) \]

\[ H_t=F\_{commit}(H\_{t-1},u\_{t,K}) \]

\[ logits_t=W\_{vocab}N(u\_{t,K}) \]

onde:

- (e_t) é o embedding do token atual;

- (H\_{t-1}) é o estado temporal disponível antes do token atual;

- (u\_{t,k}) é o State Packet após (k) passos no Capability Graph;

- (v_k) é o node executado no passo (k);

- (\_{v_k}) é a transformação residual daquele node;

- (F\_{commit}) decide como o resultado final altera o estado temporal;

- \(N\) representa a normalização final antes da projeção para o vocabulário.

Essa formulação deixa claro que o Capability Graph e o mecanismo temporal são problemas diferentes.

## 6.2 Requisito de treinamento

O mecanismo escolhido para (F) precisa oferecer estratégia de treinamento paralelizável ou suficientemente eficiente.

Candidatos para benchmark:

- SSM causal;

- recorrência moderna com scan paralelo;

- linear attention;

- Transformer causal local/compacto;

- arquitetura híbrida;

- mecanismo próprio da GSNA.

A escolha não deve ser feita por preferência teórica.

Cada candidato deve ser comparado por:

- validation loss;

- throughput de treino;

- FLOPs/token;

- bytes de estado por sequência;

- latência de decode;

- estabilidade de gradiente;

- capacidade de receber commits do Capability Graph;

- qualidade quando o Capability Graph está desativado;

- qualidade de routing produzido a partir de seu estado.

## 6.3 Hipótese de implementação inicial

Para reduzir risco no primeiro protótipo, a arquitetura inicial do FSF deve priorizar:

1.  causalidade simples de verificar;

2.  estado de inferência limitado;

3.  operações altamente batchable;

4.  shape estável para todos os Capability Nodes;

5.  possibilidade de substituir o mecanismo temporal sem reescrever o runtime do grafo.

O primeiro benchmark deve tratar o FSF como interface, não como dogma arquitetural.

Um contrato possível:

    TemporalState -> FSF.read(token_embedding) -> StatePacket
    StatePacket   -> Capability Graph          -> StatePacket'
    StatePacket'  -> FSF.commit(...)           -> TemporalState'

Se essa interface permanecer estável, SSM, recorrência, atenção local ou outro mecanismo podem ser testados mantendo o restante da GSNA constante.

## 6.4 Orçamento de comunicação

Uma das hipóteses fundamentais é que mover estado seja muito mais barato do que mover pesos.

Se o vetor principal do State Packet possuir dimensão (d) e usar (p) bytes por elemento:

\[ StateBytes d p \]

Por exemplo, para (d=1024) em BF16:

\[ StateBytes = 2048 \]

aproximadamente 2 KiB para o vetor principal.

Isso deve ser comparado a Capability Nodes que podem ocupar megabytes.

Portanto, uma métrica nova deve ser registrada:

\[ StateTransportBytes/token \]

Ela mede quanto tráfego neural de estado é necessário para navegar o grafo, separadamente de:

\[ ParameterBytesFetched/token \]

O projeto só ganha se ambos permanecerem controlados.

# 7. Capability Graph

O Capability Graph é um grafo de transformações neurais endereçáveis.

Um node (v_i) possui parâmetros (\_i) e transforma estado.

A forma preferida continua residual:

\[ h’ = h + \_i(h;\_i) \]

Isso permite:

- degradação graciosa;

- especialização incremental;

- composição;

- menor dependência de uma ordem fixa;

- possibilidade de congelar capabilities estáveis.

## 7.1 Direct edges

A v0.3 torna explícito que edges podem representar transições reais de compute.

Portanto, uma trajetória pode ser:

    Language
       |
       v
    Portuguese
       |
       v
    Brazil
      /   \
     v     v
    Law  Geography

Não existe requisito de retorno ao RNC após cada node.

O RNC pode participar da decisão, mas a topologia local pode oferecer transições diretas.

## 7.2 Topologia local + busca hierárquica

Para evitar comparação contra milhões de nodes, a política deve combinar duas escalas:

1.  **local adjacency** — vizinhos semanticamente/topologicamente plausíveis;

2.  **hierarchical routing** — acesso a regiões distantes do grafo.

Assim, o próximo node pode ser escolhido entre:

\[ Candidates(v_k)=Neighbors(v_k)GatewayCandidates(h_k) \]

A hipótese é que conhecimento relacionado forme regiões com alta localidade de rota e, consequentemente, alta localidade física.

# 8. Routing

O roteamento é uma das áreas que ainda não foi resolvida pelas aplicações predecessoras.

O Eyle deliberadamente não implementa semantic routing no registry. O MMB recebe IDs já escolhidos pelo router do modelo original.

Na GSNA, o router precisa ser treinado.

Uma formulação inicial:

\[ score(ij) = s\_{semantic}(h_i,k_j) + s\_{topology}(i,j) + s\_{reuse}(j) - c\_{physical}(j,B) \]

onde:

- (k_j): routing key compacta;

- (s\_{semantic}): compatibilidade entre estado e capability;

- (s\_{topology}): prior da aresta;

- (s\_{reuse}): valor de reutilizar node residente/quente;

- (c\_{physical}): custo estimado de acessar o node;

- (B): budget.

O roteamento não pode otimizar custo físico a ponto de trocar uma capacidade necessária por uma errada.

Portanto, custo deve atuar como preferência entre trajetórias semanticamente aceitáveis, não como substituto da qualidade.

## 8.1 Routing collapse

A arquitetura precisa prevenir o ciclo:

    node recebe mais tráfego
            |
            v
    recebe mais gradiente
            |
            v
    fica melhor
            |
            v
    recebe ainda mais tráfego

Estratégias candidatas:

- soft routing no warmup;

- top-k com straight-through estimator;

- Gumbel-style exploration;

- auxiliary load-balance loss;

- entropy floor;

- minimum exploration quota;

- per-node capacity;

- utilization regularization;

- router temperature schedule.

A escolha exata é uma decisão bloqueante para o protótipo.

# 9. Frontier-Scheduled Graph Execution

Trajetórias dinâmicas criam um problema: GPUs são eficientes com grandes batches; grafos token-específicos tendem a fragmentar execução.

A v0.3 propõe um mecanismo de execução inspirado pela experiência de batching e page-major scheduling do MMB:

**os State Packets mantêm rotas individuais, mas o runtime agrupa a fronteira corrente por node.**

Exemplo:

    packet 1 -> Node 81
    packet 2 -> Node 712
    packet 3 -> Node 81
    packet 4 -> Node 81
    packet 5 -> Node 712

O executor transforma a frontier em:

    Node 81  <- packets 1,3,4
    Node 712 <- packets 2,5

e executa:

    route frontier
          |
          v
    group by node_id
          |
          v
    pin/load required nodes
          |
          v
    batched node compute
          |
          v
    update packet states
          |
          v
    next frontier

Esse mecanismo tenta preservar simultaneamente:

- roteamento por estado;

- batching;

- locality;

- reuse;

- page-major execution;

- menor número de page faults.

Ele é uma proposta nova da v0.3 e precisa ser validado.

# 10. Integration

Uma trajetória pode ser linear ou possuir fan-out.

Para o primeiro protótipo, a v0.3 recomenda limitar fan-out e utilizar integração residual ponderada:

\[ h\_{k+1} = h_k+ \_{iA_k} \_i_i(h_k) \]

com:

\[ \_i_i=1 \]

Isso permite top-k paralelo sem introduzir imediatamente um Integration Node complexo em cada bifurcação.

Integração recursiva, message passing assíncrono e múltiplos ramos independentes devem permanecer fora do primeiro protótipo.

# 11. Memory Graph na GSNA

O Memory Graph continua distinto do Capability Graph.

## Parameter / Capability Graph

Representa capacidade neural aprendida:

    language transformations
    reasoning primitives
    code capability
    mathematics
    domain representations
    specialized neural computation

## Memory Graph

Representa conhecimento explicitamente revisável:

    facts
    events
    observations
    hypotheses
    relations
    provenance
    temporal validity
    revision
    confidence

A regra permanece:

    generalizable capability
        -> Parameter Graph

    mutable/verifiable knowledge
        -> Memory Graph

## 11.1 Memory Gateway

O ponto ainda não resolvido é a interface neural.

A v0.3 propõe um **Memory Gateway** que converta uma consulta do State Packet em uma operação de memória e retorne uma representação compacta.

A primeira implementação pode continuar materializando conteúdo estruturado e codificá-lo neuralmente.

Uma evolução posterior pode retornar embeddings, typed records ou estados neurais sem serialização textual completa.

O Memory Gateway deve preservar os invariantes do Eyle:

- identidade;

- proveniência;

- revisão;

- temporalidade;

- bounded materialization;

- ausência de hidden global semantic projection.

# 12. Runtime GSNA

O runtime deve ser tratado como parte da arquitetura.

Uma composição mínima:

    GSNA Runtime
    |
    +-- Node Registry
    +-- Graph Topology Store
    +-- Frontier Scheduler
    +-- Residency Manager
    +-- RAM Cache
    +-- VRAM Cache
    +-- Parameter Store
    +-- Lease Manager
    +-- Prefetch Queue
    +-- Integrity Verifier
    +-- Budget Manager
    +-- Telemetry
    +-- Failure / Recovery

## 12.1 Contratos herdados do MMB

A v0.3 recomenda manter os seguintes invariantes:

1.  node ativo possui lease/pin durante compute;

2.  node corrompido não produz compute silencioso;

3.  page length, shape, type e version precisam coincidir;

4.  load de batch deve possuir commit consistente;

5.  métricas ausentes permanecem unavailable;

6.  runtime não altera semanticamente a rota sem sinalizar failure;

7.  cache policy é substituível e mensurável;

8.  I/O e compute possuem janelas de telemetria separáveis.

## 12.2 Diferença para o MMB

O MMB materializa experts escolhidos por uma arquitetura fixa.

A GSNA precisa materializar nodes escolhidos por uma política treinada com feedback sobre:

- residency;

- bytes fetched;

- cache hit;

- page faults;

- latency;

- prefetch accuracy;

- compute budget.

Essa realimentação é o passo de co-design que o MMB, por construção, não poderia introduzir no modelo original.

# 13. Budget-Constrained Training

O treinamento da GSNA deve otimizar qualidade sob restrições físicas explícitas.

A ideia não é treinar primeiro ignorando hardware e depois adicionar um runtime que tente “consertar” o padrão de acesso.

O hardware disponível faz parte do ambiente de treinamento.

Definimos um vetor de budget:

\[ B_t = ( B\_{ram}, B\_{vram}, B\_{io}, B\_{compute}, B\_{latency}, B\_{prefetch} ) \]

onde:

- (B\_{ram}): quantidade de RAM que o runtime pode consumir;

- (B\_{vram}): quantidade de VRAM disponível para compute/cache;

- (B\_{io}): limite de bytes ou bandwidth para materialização;

- (B\_{compute}): limite de FLOPs ou passos de grafo;

- (B\_{latency}): restrição de tempo para a operação;

- (B\_{prefetch}): limite de bytes que podem ser carregados especulativamente.

Quando o documento usa uma forma mais curta:

\[ C=C(B,Q,L) \]

os símbolos significam:

- \(B\) = **orçamento físico** disponível para a configuração;

- \(Q\) = **qualidade desejada**, expressa por uma ou mais métricas de tarefa;

- \(L\) = **restrição de latência** aceitável.

Essa forma curta serve para discutir sizing do RNC. Ela não substitui a telemetria detalhada de (B_t).

Uma função objetivo genérica permanece:

\[ L = L\_{LM} +*rL*{routing} +*uL*{utilization} +*cL*{compute} +*lL*{locality} +*pL*{paging} +*iL*{io} +*sL*{steps} +*bL*{budget} +*fL*{prefetch} \]

onde:

- (L\_{LM}): loss autoregressiva principal;

- (L\_{routing}): qualidade/regularização do roteamento;

- (L\_{utilization}): evita concentração degenerada em poucos nodes;

- (L\_{compute}): penaliza compute desnecessário;

- (L\_{locality}): favorece reuso quando semanticamente aceitável;

- (L\_{paging}): penaliza page faults e materializações caras;

- (L\_{io}): penaliza bytes transferidos;

- (L\_{steps}): controla profundidade dinâmica;

- (L\_{budget}): penaliza violações do orçamento físico;

- (L\_{prefetch}): treina previsibilidade sem permitir que prefetch altere semântica.

Nem todos os termos devem ser ativados no primeiro treinamento.

A ordem correta é:

1.  provar que o Graph LM aprende linguagem;

2.  provar que o routing não colapsa;

3.  provar que capacidade adicional melhora qualidade;

4.  só então pressionar locality, paging e budgets.

Uma penalidade física forte demais no início pode ensinar o modelo a simplesmente não usar suas capabilities.

## 13.1 Hard constraints versus soft penalties

Alguns limites devem ser tratados como restrições duras:

    max_graph_steps
    max_active_nodes
    max_resident_bytes
    max_prefetch_bytes

Outros podem ser penalties suaves:

    reuse preference
    routing entropy
    average FLOPs
    average bytes/token

Essa distinção precisa ser explícita no runtime e no treinamento.

## 13.2 Resource Dropout revisado

Resource Dropout continua válido, mas não deve transformar o menor hardware em objetivo universal.

O mesmo checkpoint pode ser exposto durante treinamento a perfis como:

    Profile A: baixo budget de RAM, storage lento
    Profile B: RAM moderada, storage rápido
    Profile C: VRAM disponível
    Profile D: grande cache residente

A meta é aprender degradação graciosa e políticas diferentes, não forçar todas as configurações a se comportarem como a menor.

# 14. Sparse Training State

O princípio da v0.2 permanece:

\[ TrainingState\_{resident}TrainingState\_{total} \]

quando somente um subconjunto de nodes participa de um batch/frontier.

Mas a v0.3 trata esse recurso como **fase posterior**, não requisito do primeiro Graph LM.

O primeiro objetivo é provar a arquitetura neural inteiramente residente.

Depois, optimizer state e gradients podem ser materializados por node.

Questões ainda abertas:

- staleness de optimizer state em nodes raros;

- momentum de módulos esparsamente atualizados;

- writeback policy;

- atomicidade de checkpoint;

- batching de updates;

- distributed optimizer state;

- relação entre route frequency e learning rate.

# 15. Prefetch treinável

O MMB provou que prefetch e paralelização não podem corrigir completamente um padrão de acesso adverso.

Na GSNA, a política pode prever:

\[ P(v\_{k+1},v\_{k+2},…,v\_{k+n}\|h_k) \]

O runtime usa isso somente como hint.

A propriedade semântica permanece:

prefetch pode errar sem mudar a resposta; routing não pode ser substituído pelo prefetch.

Métricas:

    prefetch_hit_rate
    prefetched_bytes/token
    blocking_bytes/token
    wasted_prefetch_bytes/token
    prefetch_lead_time

# 16. Módulos como áreas de conhecimento

Capability Nodes podem desenvolver especializações como:

- sintaxe;

- português;

- código;

- álgebra;

- planejamento;

- geometria;

- medicina;

- entidades;

- ferramentas;

- abstração.

Esses nomes são interpretações posteriores.

A arquitetura não deve depender de uma taxonomia humana completa.

O que precisa emergir durante o treinamento é:

- reutilização;

- especialização útil;

- baixa interferência;

- rotas estáveis;

- locality;

- redução de loss.

Metadados interpretáveis podem ser derivados depois para diagnóstico, distribuição e governança.

# 17. O que a v0.3 congela

A versão 0.3 considera os pontos abaixo parte da arquitetura, e não mais perguntas abertas:

1.  **estado neural, e não token ID bruto, percorre o grafo;**

2.  **Memory Graph e Parameter/Capability Graph são distintos;**

3.  **capabilities são endereçáveis;**

4.  **capabilities podem possuir parâmetros não residentes;**

5.  **Resident Neural Core (RNC) substitui o conceito de “RNC pequeno” como terminologia canônica;**

6.  **o RNC é residente, mas não é hub obrigatório entre capabilities;**

7.  **o RNC pode crescer quando isso melhora qualidade; seu tamanho não é constante por definição;**

8.  **o tamanho do RNC e o tamanho do Capability Graph são eixos de scaling arquiteturalmente desacoplados;**

9.  **RNC-128MB é o baseline inicial de pesos residentes, não um limite permanente;**

10. **Fast State Fabric é componente de primeira classe e pertence logicamente ao RNC;**

11. **direct capability-to-capability transitions são permitidas;**

12. **roteamento deve considerar semântica, topologia e custo físico;**

13. **runtime não é semantic router;**

14. **paging é parte esperada da arquitetura, não fallback;**

15. **bytes/token é métrica de primeira classe;**

16. **StateTransportBytes/token deve ser medido separadamente de ParameterBytesFetched/token;**

17. **o primeiro Graph LM deve ser testado inteiramente residente antes de paging físico;**

18. **frontier scheduling é o modelo de execução candidato para preservar batching sem destruir rotas individuais;**

19. **crescimento estrutural não é requisito do primeiro protótipo;**

20. **continual learning, multiresolução e distribuição ficam depois da prova de routing/capacity scaling;**

21. **o target de hardware é um parâmetro de deployment, não uma definição da arquitetura.**

# 18. Registro de decisões em aberto

Esta seção é o principal backlog científico e de engenharia da v0.3.

Uma decisão é marcada como:

- **FROZEN** — congelada para o baseline inicial;

- **OPEN / BLOCKING** — precisa ser resolvida antes do primeiro experimento correspondente;

- **OPEN / NON-BLOCKING** — pode usar baseline simples;

- **DEFERRED** — deliberadamente adiada.

## D-001 — Orçamento do RNC inicial

**Status:** FROZEN para o baseline.

\[ RNCWeightBytes = 128 \]

A primeira implementação deve usar esse budget como ponto central.

Ainda precisa registrar:

- precisão dos pesos;

- número efetivo de parâmetros;

- embeddings incluídos ou não;

- LM head incluído ou não;

- overhead real;

- RNCTotalResidentBytes.

## D-002 — Precisão e número de parâmetros do RNC-128MB

**Status:** OPEN / BLOCKING.

O budget é em bytes. A precisão ainda não está congelada.

Candidatos iniciais:

- BF16/FP16 para baseline de treinamento simples;

- FP32 somente para debug/ablation;

- INT8/4-bit principalmente para inferência e testes posteriores.

Pergunta:

é melhor um RNC com menos parâmetros e maior precisão ou mais parâmetros com quantização, sob o mesmo budget de 128 MB?

Teste obrigatório: comparar qualidade e throughput sob budget idêntico de bytes.

## D-003 — Arquitetura interna do RNC

**Status:** OPEN / BLOCKING.

O RNC precisa oferecer inteligência basal e um estado adequado ao roteamento.

Candidatos:

- SSM causal;

- recorrência moderna com scan;

- atenção causal local;

- pequeno Transformer;

- híbrido SSM + atenção local;

- mecanismo próprio.

A melhor escolha será a que maximizar:

\[ \]

sem transformar o RNC em hub obrigatório.

## D-004 — Arquitetura do Fast State Fabric

**Status:** OPEN / BLOCKING.

Definir:

- dimensão do estado;

- número de streams de estado;

- mecanismo temporal;

- forma de read;

- forma de commit;

- como múltiplas branches escrevem de volta;

- state reset;

- estado por sequência durante batching.

O FSF pode compartilhar componentes com o backbone do RNC.

## D-005 — Forma do State Packet

**Status:** OPEN / BLOCKING.

Candidatos:

- vetor \[d\];

- pequeno conjunto \[m,d\];

- vetor + recurrent state;

- vetor + janela local limitada;

- múltiplos canais tipados.

O State Packet precisa ser pequeno o bastante para transporte barato e rico o bastante para não se tornar gargalo informacional.

## D-006 — Primitive de Capability Node

**Status:** OPEN / BLOCKING.

Baseline recomendado para a primeira implementação:

    Norm
     -> Linear / gated expansion
     -> activation
     -> projection
     -> residual

ou equivalente SwiGLU residual.

Razões:

- shape uniforme;

- fácil batching;

- gradiente simples;

- implementação eficiente;

- permite medir o valor do grafo antes de introduzir nodes heterogêneos.

Depois podem ser testados:

- tiny SSM;

- tiny attention;

- low-rank specialists;

- convolutional/gated nodes;

- multimodal nodes.

## D-007 — Granularidade de node/page

**Status:** OPEN / NON-BLOCKING para Graph LM residente; BLOCKING para paging.

Testar faixas de parâmetros e bytes.

A unidade neural e a página física não precisam ser idênticas.

Questões:

- um node pode ocupar múltiplas pages?

- múltiplos nodes podem compartilhar uma page?

- qual é o custo de metadata?

- qual o desperdício por miss?

- qual o efeito sobre batching?

## D-008 — Algoritmo de routing diferenciável

**Status:** OPEN / BLOCKING.

Baseline recomendado para testar primeiro:

1.  soft routing no warmup;

2.  annealing de temperatura;

3.  top-k esparso;

4.  auxiliary load-balance loss;

5.  entropy floor;

6.  exploration controlada;

7.  hard cap de utilização por frontier.

Ainda precisa decidir como o gradiente atravessa a seleção discreta.

Candidatos:

- straight-through estimator;

- Gumbel-style relaxation;

- routing supervision auxiliar;

- distillation de soft router para hard router.

## D-009 — Granularidade temporal do routing

**Status:** OPEN / BLOCKING.

A semântica lógica preferida é **token-state specific routing**.

Entretanto, execução física pode agrupar muitos State Packets que escolheram o mesmo node.

Isso separa:

    logical routing granularity = por estado/token
    physical batching           = por frontier/node

Testes devem comparar routing por:

- token;

- microchunk;

- chunk;

- híbrido.

## D-010 — Edge semantics

**Status:** OPEN / BLOCKING.

A v0.3 favorece duas classes de arestas:

1.  **local direct edges**, usadas para transições frequentes entre capabilities relacionadas;

2.  **global gateway transitions**, usadas para saltar para regiões distantes.

Uma formulação candidata:

\[ Candidates(v_k)=Neighbors(v_k)Gateways(h_k) \]

Isso reduz a necessidade de consultar um router global contra todos os nodes a cada passo.

## D-011 — Router local versus router global

**Status:** OPEN / BLOCKING.

Uma implementação candidata:

    State Packet
       |
       +--> local route head on current node
       |
       +--> global route head on RNC/route index when needed

O objetivo é que a maioria das transições maduras seja local e previsível, usando o roteamento global apenas quando necessário.

Pergunta:

que fração de steps pode permanecer em rotas locais sem perder capacidade de recombinação global?

## D-012 — Fan-out e integração

**Status:** OPEN / BLOCKING.

Primeiro baseline recomendado:

- top-k pequeno;

- máximo de 2 branches em paralelo;

- weighted residual merge;

- hard max de fan-out.

A integração pode começar como:

\[ h’ = h + \_{i A} w_i_i(h) \]

com:

\[ \_i w_i = 1 \]

Depois testar integração sequencial, learned merge e Integration Nodes especializados.

## D-013 — Stop / continue

**Status:** OPEN / NON-BLOCKING para primeira versão.

Primeiro protótipo pode usar número máximo fixo de graph steps.

Depois introduzir:

\[ P(stop\|h_k) \]

com:

- hard max;

- compute penalty;

- threshold;

- minimum steps;

- target de estabilidade.

## D-014 — Commit para Fast Temporal State

**Status:** OPEN / BLOCKING.

Esse é um dos contratos mais importantes.

É necessário decidir quais informações produzidas pelo Capability Graph alteram (H_t), o estado usado pelos tokens futuros.

Questões:

- commit somente após terminar a trajetória?

- commits intermediários são permitidos?

- branches podem escrever em paralelo?

- existe gating de commit?

- como evitar overwrite destrutivo?

- como impedir leakage de informação futura em treino paralelo?

## D-015 — Causalidade formal

**Status:** OPEN / BLOCKING.

Precisamos provar por testes e invariantes que:

\[ logits_t x\_{t+1:} \]

Nenhuma otimização de frontier, grouping, Memory Graph, cache ou training batch pode quebrar essa propriedade.

O repositório deve ter testes de causal masking/leakage desde o primeiro protótipo.

## D-016 — Shared embeddings e LM head

**Status:** OPEN / BLOCKING para fechar o budget de 128 MB.

Decidir:

- embeddings fazem parte do RNC?

- LM head fica residente?

- input embedding e output projection compartilham pesos?

- vocabulário grande consome budget demais?

- projeção do vocabulário pode usar estratégia fatorada?

Essa decisão pode consumir grande fração dos 128 MB e precisa ser tratada cedo.

## D-017 — Hierarchical router

**Status:** OPEN / NON-BLOCKING em escala pequena.

O protótipo pode usar router flat.

Quando nodes crescerem para milhares ou milhões, será necessário:

    domain / region
       -> cluster
          -> local neighborhood
             -> node

O custo do índice de routing também precisa ser contado em bytes residentes.

## D-018 — RNC-only fallback

**Status:** OPEN / NON-BLOCKING.

Definir comportamento quando:

- graph budget é zero;

- node solicitado está indisponível;

- page falha integridade;

- latência excede budget;

- storage está indisponível.

Preferência: fail-closed para corrupção e degradação semântica explícita para ausência de capacidade, sem substituição silenciosa de node.

## D-019 — Layout físico

**Status:** OPEN / DEFERRED até prova residente.

Definir:

- node-to-file mapping;

- shard size;

- adjacency-aware packing;

- compaction;

- versioning;

- checksum granularity;

- atomic replacement.

Os invariantes de integridade devem seguir a disciplina já demonstrada no MMB.

## D-020 — Cache policy

**Status:** OPEN / DEFERRED até prova residente.

LRU deve ser baseline.

Depois comparar:

- LRU;

- LFU;

- reuse-distance aware;

- topology-aware;

- route-predicted residency.

Uma política sofisticada só se justifica se telemetria provar benefício.

## D-021 — Prefetch horizon

**Status:** OPEN / DEFERRED até routing estável.

Definir:

- quantos steps futuros prever;

- quantos bytes máximos especular;

- priority;

- cancelamento;

- tratamento de branches.

Prefetch nunca altera semântica. Ele somente antecipa materialização.

## D-022 — Sparse optimizer

**Status:** OPEN / DEFERRED.

Definir:

- materialização de optimizer state;

- persistence;

- staleness;

- writeback;

- batching de updates;

- learning rate para nodes raros;

- checkpoint atômico.

## D-023 — Memory Gateway

**Status:** OPEN / NON-BLOCKING para Graph LM inicial.

Definir interface neural entre State Packet e o Memory Graph herdando:

- identity;

- provenance;

- revision;

- temporal validity;

- bounded recall.

O objetivo é evitar converter toda memória recuperada obrigatoriamente em texto.

## D-024 — Structural growth

**Status:** DEFERRED.

Spawn, split, merge, prune, freeze e retire só entram depois que uma topologia fixa demonstrar capacity scaling.

## D-025 — Multiresolution nodes

**Status:** DEFERRED.

Base + refinements permanece interessante para hardware adaptativo, mas aumenta cedo demais o espaço experimental.

## D-026 — Distributed GSNA

**Status:** DEFERRED.

Primeiro provar em uma máquina.

Depois decidir network placement, remote nodes, replication e consistency.

## D-027 — Curva de Core Scaling

**Status:** OPEN / REQUIRED EXPERIMENT.

Treinar RNCs sob diferentes budgets, por exemplo:

    64 MB
    128 MB
    256 MB
    512 MB

mantendo o Capability Graph fixo.

Pergunta:

quanto de qualidade, routing e integração é ganho por byte residente adicional?

## D-028 — Curva de Capability Scaling

**Status:** OPEN / REQUIRED EXPERIMENT.

Fixar inicialmente:

    RNC = 128 MB

e escalar o grafo:

    1x
    2x
    4x
    8x

mantendo approximately constantes:

- active nodes;

- active FLOPs;

- max graph steps;

- RNC budget.

Pergunta:

capacidade total adicional melhora qualidade sem exigir compute ativo proporcional?

## D-029 — Matriz 2D Core × Graph

**Status:** OPEN / REQUIRED EXPERIMENT após D-027/D-028.

Medir a interação:

| **RNC budget  Capability Graph** | **1x** | **2x** | **4x** | **8x** |
|----------------------------------|--------|--------|--------|--------|
| 64 MB                            | teste  | teste  | teste  | teste  |
| **128 MB**                       | teste  | teste  | teste  | teste  |
| 256 MB                           | teste  | teste  | teste  | teste  |
| 512 MB                           | teste  | teste  | teste  | teste  |

Para cada célula registrar:

- validation loss;

- downstream quality;

- routing accuracy/proxy;

- routing entropy;

- route stability;

- node utilization;

- active params/token;

- FLOPs/token;

- state bytes/token;

- resident bytes;

- tokens/s.

Essa matriz responde se existe uma relação como:

G_usable = f(C)

ou se o Capability Graph continua melhorando qualidade quase independentemente do RNC dentro da faixa estudada.

## D-030 — Hardware target mínimo

**Status:** OPEN / NON-BLOCKING.

A arquitetura não deve ser definida por uma máquina específica.

Para o protótipo, é útil escolher uma classe de referência — por exemplo, máquina com 8 GB de RAM — apenas para medir se o RNC-128MB deixa working set suficiente para cache, runtime e estado.

O target é benchmark, não dogma.

# 19. Minimal Graph Language Model — proposta para o primeiro experimento

A v0.3 reduz o primeiro protótipo ao mínimo capaz de falsificar a hipótese neural, mas agora parte explicitamente do **RNC-128MB**.

## 19.1 Componentes

    Tokenizer
       |
    Embedding
       |
    +-----------------------------+
    | Resident Neural Core        |
    | RNCWeightBytes = 128 MB     |
    |                             |
    | basal neural capacity       |
    | Fast State Fabric           |
    | routing support             |
    | integration / commit        |
    +--------------+--------------+
                   |
              State Packets
                   |
                   v
              Router / edges
                   |
                   v
           Capability Graph
           /       |       \
          v        v        v
        Node A -> Node B -> Node C
           \               /
            +-------------+
                   |
                   v
           Integration / Commit
                   |
                   v
                LM Head

No primeiro protótipo não entram:

- paging físico;

- Memory Graph;

- structural growth;

- multiresolution;

- continual learning;

- distributed runtime;

- sparse optimizer paging.

A intenção é isolar a hipótese neural.

## 19.2 Baseline recomendado

Congelado:

    RNC weights budget:      128 MB
    routing logic:           token-state specific
    physical execution:      frontier grouped
    Capability Node shape:   uniforme no baseline
    paging:                  disabled
    Memory Graph:            disabled

Ainda a decidir por D-002..D-016:

    RNC precision
    effective RNC params
    hidden dimension
    Fast State architecture
    State Packet shape
    node parameter count
    number of nodes
    top-k
    graph steps
    embedding/head strategy

## 19.3 Arquitetura de Capability Node baseline

Para reduzir variáveis, o primeiro baseline deve usar uma transformação residual uniforme e simples.

Exemplo conceitual:

\[ z = Norm(h) \]

\[ u = SwiGLU(W_1z,W_gz) \]

\[ (h)=W_2u \]

\[ h’ = h + \_i(h) \]

onde (\_i) pode ser um gate aprendido ou peso de routing.

A escolha exata de ativação pode mudar, mas o princípio é:

o primeiro experimento deve medir o valor do **grafo e do routing**, não o valor de uma família exótica de nodes.

## 19.4 Routing baseline

Fluxo candidato:

    warmup
      -> soft probabilities over candidates
      -> high exploration

    anneal
      -> lower temperature
      -> load-balance regularization

    sparse phase
      -> top-k active nodes
      -> frontier batching

Durante toda a transição medir collapse, utilização e estabilidade.

## 19.5 Objetivo

Responder apenas:

**Um RNC residente de 128 MB consegue produzir estado causal suficiente para navegar por capabilities neurais dinamicamente roteadas e aprender next-token prediction com qualidade competitiva sob compute ativo comparável?**

E, em seguida:

**aumentar o Capability Graph melhora qualidade mantendo RNC, active nodes e FLOPs aproximadamente constantes?**

Se a segunda resposta for negativa, paging não resolve o problema e não deve ser a próxima prioridade.

# 20. Programa experimental 0.3

O programa experimental deve separar claramente **viabilidade neural**, **Core Scaling**, **Capability Scaling** e somente depois **eficiência física**.

## Fase A — RNC-128MB Language Baseline

Treinar o RNC de 128 MB sem Capability Graph ativo ou com graph mínimo.

Objetivo:

estabelecer a qualidade basal do núcleo residente.

Medir:

- validation loss;

- perplexity;

- tokens/s;

- FLOPs/token;

- RNCWeightBytes;

- RNCTotalResidentBytes;

- state bytes/sequence;

- decode latency.

Essa fase responde quanto da qualidade vem do RNC antes de atribuir ganhos ao grafo.

## Fase B — Resident Graph LM

Adicionar Capability Graph inteiramente residente.

Medir:

- validation loss;

- routing entropy;

- node utilization;

- route stability;

- steps/token;

- tokens/s;

- active params/token;

- train FLOPs;

- StateTransportBytes/token.

Baselines mínimos:

    RNC-only
    Dense Transformer
    MoE Transformer
    SSM/recurrent baseline
    GSNA

O objetivo é provar que o roteamento dinâmico não destrói modelagem de linguagem.

## Fase C — Capability Scaling com RNC fixo

Fixar:

    RNCWeightBytes = 128 MB

Treinar:

    GSNA-1x
    GSNA-2x
    GSNA-4x
    GSNA-8x

mantendo approximately constantes:

- active nodes;

- graph steps;

- active FLOPs/token;

- State Packet shape;

- RNC budget.

Pergunta:

**Aumentar capabilities disponíveis melhora qualidade sem exigir aumento proporcional de compute ativo?**

Essa é a primeira prova estrutural.

## Fase D — Core Scaling com grafo fixo

Fixar um Capability Graph e variar:

    64 MB
    128 MB
    256 MB
    512 MB

Perguntas:

1.  quanto de qualidade vem de aumentar inteligência residente?

2.  routing melhora com core maior?

3.  route stability melhora?

4.  capabilities externas passam a ser melhor utilizadas?

5.  em que ponto o custo residente deixa de compensar?

Métrica conceitual:

\[ CoreScalingEfficiency = {RNCWeightBytes+ActiveFLOPs+Latency} \]

## Fase E — Matriz 2D Core × Graph

Executar a matriz:

| **RNC budget  Graph** | **1x** | **2x** | **4x** | **8x** |
|-----------------------|--------|--------|--------|--------|
| 64 MB                 | ●      | ●      | ●      | ●      |
| **128 MB**            | ●      | ●      | ●      | ●      |
| 256 MB                | ●      | ●      | ●      | ●      |
| 512 MB                | ●      | ●      | ●      | ●      |

A matriz deve usar seeds e budgets de treino controlados.

Perguntas:

- um RNC pequeno deixa de aproveitar graphs grandes?

- um graph grande compensa um RNC menor?

- existe fronteira de Pareto entre bytes residentes e capacidade externa?

- quais pares oferecem melhor qualidade por custo?

- existe interação forte G_usable = f(C)?

O resultado esperado não é uma resposta universal. É um mapa experimental.

## Fase F — Routing Granularity

Comparar:

    per-token state routing
    microchunk routing
    chunk routing
    hybrid routing

sem alterar o resto da arquitetura.

Medir:

- quality;

- route diversity;

- batching efficiency;

- tokens/s;

- steps/token;

- frontier fragmentation.

## Fase G — Simulated Residency Budget

Ainda sem SSD real.

Simular diferentes budgets para Capability Nodes:

    small
    medium
    large
    full resident

Introduzir:

- cache state;

- locality cost;

- page miss penalty;

- Resource Dropout.

Não confundir esse budget com os 128 MB de pesos do RNC.

Pergunta:

**o mesmo checkpoint aprende trajetórias diferentes sob budgets físicos diferentes sem colapsar qualidade?**

## Fase H — Physical Paging

Somente depois das fases anteriores.

Reutilizar princípios já demonstrados no MMB:

- parameter store;

- checksums;

- leases;

- cache;

- pin;

- atomic commit;

- telemetry;

- fail-closed.

Medir:

    ParameterBytesFetched/token
    blocking_bytes/token
    StateTransportBytes/token
    cache hit ratio
    evictions
    page faults
    prefetched bytes/token
    latency

## Fase I — Predictive Prefetch

Adicionar previsão de futuras capabilities.

Objetivo:

transformar rota semanticamente estável em rota fisicamente previsível.

## Fase J — Sparse Training State

Paginar weights, gradients e optimizer state de nodes não ativos.

A prova é:

\[ TrainingState\_{resident}TrainingState\_{total} \]

sem perda catastrófica de throughput ou convergência.

## Fase K — Memory Graph

Integrar Eyle Memory Graph por Memory Gateway.

Testar:

- factual recall;

- temporal update;

- contradiction;

- provenance;

- revision;

- long-horizon memory;

- bounded active memory.

## Fase L — Structural Growth

Somente após as fases anteriores:

- spawn;

- split;

- merge;

- prune;

- freeze;

- retire.

# 21. Gates de realidade

A GSNA só avança de fase quando cumprir gates explícitos.

## Gate 1 — Language

GSNA aprende next-token prediction sem depender de uma pilha Transformer completa como caminho principal.

## Gate 2 — Routing

Não existe routing collapse destrutivo e a utilização efetiva de nodes permanece mensurável.

## Gate 3 — Capacity

Aumentar capacidade total melhora validation loss mantendo active compute aproximadamente estável.

## Gate 4 — Locality

Sob budget simulado, a rede reduz miss cost/bytes sem sacrificar desproporcionalmente qualidade.

## Gate 5 — Physical Paging

Grande parte do Parameter Graph pode ficar não residente com degradação controlada.

## Gate 6 — Budget Adaptation

O mesmo modelo opera sob múltiplos budgets e degrada gradualmente.

## Gate 7 — Sparse Training State

Estado total de treinamento excede o estado residente sem inviabilizar convergência.

## Gate 8 — Memory

Memory Graph cresce sem crescimento proporcional do estado neural/contextual ativo.

# 22. Métricas obrigatórias

## Qualidade

    validation loss
    perplexity
    reasoning
    coding
    factuality
    memory tasks

## Roteamento

    routing entropy
    node utilization
    top-k concentration
    route stability
    route depth
    branch count
    router confidence

## Compute

    FLOPs/token
    active params/token
    steps/token
    tokens/s
    batch occupancy
    frontier fragmentation

## Residência

    resident parameters
    resident bytes
    peak resident bytes
    active/resident ratio

## I/O

    logical bytes/token
    physical bytes/token
    blocking bytes/token
    prefetched bytes/token
    page faults/token

## Cache

    hit ratio
    reuse distance
    evictions
    hotset size

## Treinamento

    optimizer resident bytes
    gradient resident bytes
    nodes updated/batch
    I/O/optimizer step
    route-frequency distribution

## Memória

    recall precision
    useful recall
    revision correctness
    provenance preservation
    temporal consistency
    active memory bytes/token

# 23. Falhas que falsificariam a hipótese

A GSNA deve definir condições sob as quais a ideia central é considerada não suportada.

A hipótese enfraquece fortemente se:

1.  aumentar o número total de nodes não melhora qualidade com active compute fixo;

2.  o router só mantém qualidade utilizando grande fração dos nodes;

3.  rotas úteis possuem baixa localidade intrínseca;

4.  o Fast State Fabric precisa crescer proporcionalmente ao Capability Graph;

5.  frontier scheduling destrói GPU utilization;

6.  paging real consome a maior parte da latência mesmo após treinamento hardware-aware;

7.  budgets pequenos provocam falha abrupta em vez de degradação gradual;

8.  especialização não emerge e os nodes convergem para redundância;

9.  o modelo precisa reintroduzir uma grande stack fixa para manter qualidade;

10. a complexidade de routing supera a economia obtida pela sparsity.

Resultados negativos nesses pontos são valiosos: eles indicam onde a hipótese de desacoplamento deixa de ser válida.

# 24. Riscos científicos centrais

## 24.1 Routing collapse

Poucos nodes dominam e os demais deixam de aprender.

Mitigações candidatas:

- soft warmup;

- load-balance loss;

- entropy floor;

- exploration;

- capacity caps;

- monitoring por frontier.

## 24.2 Router search problem

A capacidade total pode crescer mais rápido que a capacidade do sistema de encontrar a região correta.

Isso pode gerar:

\|Gₚ\| ↑ sem Quality ↑

O hierarchical router, local edges e routing keys tentam atacar esse problema, mas a curva real precisa ser medida.

## 24.3 RNC capacity mismatch

O risco não é “o RNC crescer e violar a arquitetura”.

O risco correto é escolher um RNC inadequado para o tamanho e a dificuldade do grafo.

RNC pequeno demais:

- estado pobre;

- routing ruim;

- integração fraca;

- graph subutilizado.

RNC grande demais:

- custo residente alto;

- compute basal excessivo;

- menor vantagem relativa do Capability Graph.

A matriz 2D Core × Graph existe precisamente para localizar essa fronteira.

## 24.4 State bottleneck

O Fast State Fabric pode comprimir contexto demais e se tornar o verdadeiro limite de capacidade.

Sinais:

- aumento do graph não melhora loss;

- aumento do State Packet melhora muito mais que adicionar nodes;

- long-context degrada cedo;

- routing fica instável em sequências longas.

## 24.5 Graph fragmentation

Muitos nodes pequenos podem criar overhead maior que o compute economizado.

Precisamos medir:

\[ RoutingCost + SchedulingCost + StateTransportCost \]

contra o compute evitado.

## 24.6 I/O domination

A rede pode aprender sparsity sem aprender locality.

Esse risco já é motivado diretamente pelos limites observados no MMB.

Métrica crítica:

\[ ParameterBytesFetched/token \]

e não apenas active parameters.

## 24.7 Serial-depth latency

Trajetórias longas criam dependência serial.

Mesmo com menos FLOPs:

\[ Latency\_{GSNA} \]

pode ser maior que um baseline altamente paralelo.

Stop/continue, fan-out limitado e edges locais precisam ser medidos contra esse risco.

## 24.8 Frontier fragmentation

Cada State Packet pode pedir um node diferente, reduzindo batching.

Frontier scheduling tenta recuperar paralelismo agrupando requests por node sem mudar a rota lógica.

## 24.9 Catastrophic routing change

Pequenas mudanças de weights podem alterar caminhos inteiros.

Possíveis mitigações:

- route consistency regularization;

- router EMA;

- slow router updates;

- distillation;

- hysteresis de route priors.

## 24.10 Sparse optimizer drift

Nodes raros deixam de acompanhar a distribuição.

Esse problema pode aparecer antes mesmo de optimizer paging.

Monitorar:

- update frequency;

- gradient age;

- effective learning rate;

- utilization.

## 24.11 RNC dominates quality

Pode ocorrer de quase todo ganho vir do aumento do RNC e o Capability Graph adicionar pouco.

Esse resultado não deve ser escondido.

Ablations obrigatórias:

    RNC only
    RNC + graph
    larger RNC with same total active compute
    smaller RNC + larger graph

Se um RNC maior vencer consistentemente um graph maior sob custo comparável, a hipótese de capability scaling precisa ser revisada.

## 24.12 Graph capacity exists but is not addressable

O sistema pode possuir milhares de nodes úteis, mas o router não consegue encontrá-los de forma confiável.

Métrica candidata:

\[ AddressableCapacityEfficiency = {TotalCapability} \]

Ela não possui unidade universal; serve como conceito para comparar incrementos de capacidade.

Também medir uma curva de saturação:

    graph size -> quality gain

Se ela achatar cedo, descobrir se o limite está no router, State Packet, node primitive ou dados de treinamento.

# 25. Estratégia de implementação

A implementação deve seguir a disciplina aprendida com Eyle e MMB:

**primeiro definir autoridade, contratos e métricas; depois otimizar gargalos medidos.**

Ordem recomendada:

    1. congelar accounting do RNC-128MB
    2. congelar D-002..D-016 bloqueantes
    3. implementar RNC-only baseline
    4. implementar Graph LM residente
    5. instrumentar routing/frontier/state transport
    6. provar Capability Scaling com RNC fixo em 128 MB
    7. medir Core Scaling
    8. executar matriz 2D Core x Graph
    9. simular budgets de residência
    10. introduzir paging real
    11. otimizar somente gargalos demonstrados
    12. integrar Memory Graph
    13. explorar sparse training state
    14. somente então crescimento estrutural

## 25.1 Arquitetura de software recomendada

Separar módulos desde o primeiro commit:

    gsna/
      model/
        rnc/
        fast_state/
        capability_node/
        router/
        integration/
        lm_head/

      graph/
        registry/
        topology/
        routing_keys/
        scheduler/

      runtime/
        resident_executor/
        telemetry/
        budget/
        paging/          # inicialmente interface/stub

      memory/
        gateway/         # inicialmente interface/stub

      training/
        losses/
        routing_schedule/
        experiments/
        checkpoints/

      tests/
        causality/
        routing/
        state/
        graph/
        accounting/

O objetivo é impedir que o primeiro protótipo misture semântica neural, runtime físico e experimentação no mesmo módulo.

## 25.2 Interfaces mínimas

### RNC

    encode(token, temporal_state) -> StatePacket
    commit(StatePacket)           -> temporal_state'
    project(StatePacket)          -> logits

### Capability Node

    forward(StatePacket) -> StatePacketDelta

### Router

    route(StatePacket, current_node, budget, topology)
        -> RouteDecision

### Scheduler

    group(RouteDecision[])
        -> NodeFrontiers

### Runtime

    resolve(node_id)
    execute(frontier)
    record(telemetry)

Esses contratos devem permanecer independentes do mecanismo concreto escolhido no primeiro benchmark.

## 25.3 Testes unitários que devem existir antes de treinar

- nenhum acesso causal ao futuro;

- identidade de node estável;

- route decision determinística quando stochasticity está desligada;

- residual node preserva shape;

- frontier grouping não altera resultado;

- integração é independente da ordem quando declarada comutativa;

- budget hard limit é respeitado;

- bytes accounting fecha;

- RNCWeightBytes não ultrapassa o baseline sem erro explícito;

- unavailable node não é substituído silenciosamente;

- checksum failure é fail-closed quando paging entrar.

## 25.4 Instrumentação desde o primeiro dia

Registrar por batch e por token:

    rnc_weight_bytes
    rnc_total_resident_bytes
    state_transport_bytes
    active_nodes
    active_params
    graph_steps
    routing_entropy
    node_utilization
    frontier_size
    route_changes
    train_flops
    tokens_per_second

Quando paging entrar:

    parameter_bytes_fetched
    blocking_bytes
    prefetch_bytes
    page_faults
    cache_hits
    cache_misses
    io_wait

A GSNA não deve repetir o erro de descobrir tarde que uma otimização não move o gargalo real.

# 26. Arquitetura-alvo consolidada

                                  GSNA
                                   |
                        Resident Neural Core
                             RNC-128MB*
              +--------------------+---------------------+
              |                    |                     |
              v                    v                     v
       basal language        Fast State Fabric     routing support
          capacity           temporal state         integration
              |                    |
              |                    v
              |               State Packets
              |                    |
              +--------------------+------------------------------+
                                                                   |
                                                                   v
                                                            Capability Graph
                                                      +------------+------------+
                                                      |            |            |
                                                      v            v            v
                                                    Node A -----> Node B -----> Node C
                                                      |            |             |
                                                      +-------> Node D <---------+
                                                                   |
                                                                   v
                                                            Integration / Commit
                                                                   |
                                          +------------------------+------------------+
                                          |                                           |
                                          v                                           v
                                    Fast State update                           Memory Gateway
                                          |                                           |
                                          v                                           v
                                       LM Head                                  Memory Graph
                                                                               provenance
                                                                               revision
                                                                               temporal state

\* RNC-128MB é o baseline inicial de RNCWeightBytes, não um limite arquitetural.

## Control and data flow

A arquitetura não separa rigidamente “control plane” e “data plane” em duas máquinas isoladas.

A distinção é funcional:

- decisões de rota, budget e materialização são **controle**;

- State Packets seguindo edges e sendo transformados são **fluxo neural de dados**.

Um node pode produzir informação para o próximo node sem voltar ao RNC.

## Physical execution

    RouteDecision / node_id
            |
            v
    Node Registry
            |
            v
    Residency Manager
       /             \
    resident       non-resident
       |               |
       |          Parameter Store
       |               |
       +-------- cache / pin / lease
                       |
                       v
                     compute
                       |
                       v
                   StatePacket'

## Hierarquia de residência esperada

    always resident
    --------------
    RNC weights
    Fast State
    routing index básico
    runtime crítico

    cached / budget dependent
    -------------------------
    Capability Nodes
    routing clusters
    specialized integration nodes

    persistent / external
    ---------------------
    cold capabilities
    optimizer state futuro
    Memory Graph storage
    historical state

O objetivo físico não é manter tudo fora da RAM.

É manter residente aquilo cujo valor por byte justifica residência.

# 27. Relação com arquiteturas existentes

## Transformer

Transformer mantém uma forte estrutura de layers e utiliza attention para comunicação entre posições.

GSNA não exige uma stack fixa de capability layers. Ela separa mecanismo temporal de navegação de capacidades.

## MoE

MoE introduz sparsity de experts, mas normalmente dentro de layers fixas.

GSNA propõe um grafo global com transições dinâmicas, profundidade variável, nodes pagináveis e orçamento físico.

## RAG

RAG recupera informação para o contexto.

GSNA distingue memória explícita de capacidade neural e pode materializar ambos seletivamente.

## GNN

O grafo da GSNA não representa necessariamente entidades externas.

Ele representa a topologia de computação disponível.

## Operating systems

A analogia com memória virtual permanece útil:

    virtual address space -> logical capability space
    pages                 -> capability nodes
    page table            -> node registry
    pager                  -> residency manager
    scheduler              -> frontier scheduler
    cache                  -> resident capability cache

A diferença é que a política neural pode aprender a produzir padrões de acesso melhores.

# 28. O que não deve ser prometido

A GSNA 0.3 não promete:

- rodar arbitrariamente muitos parâmetros com RAM constante;

- custo estritamente constante em escala infinita;

- substituir Transformers em todas as tarefas;

- tornar SSD equivalente a VRAM;

- criar especialização perfeita automaticamente;

- continual learning sem forgetting;

- crescimento estrutural estável;

- treinamento de trilhões de parâmetros em hardware doméstico.

A afirmação científica é mais restrita:

**Existe a possibilidade de que capacidade total cresça mais rapidamente que compute ativo, memória residente e bandwidth por token quando a arquitetura é treinada para endereçar seletivamente capacidades e memória.**

Essa possibilidade precisa ser medida.

# 29. Primeiro marco realmente convincente

Um resultado estrutural forte seria:

    GSNA-1x
    total capacity      100M
    active parameters    25M
    resident parameters 100M
    validation loss       L1

    GSNA-2x
    total capacity      200M
    active parameters   ~25M
    validation loss     < L1

    GSNA-4x
    total capacity      400M
    active parameters   ~25M
    validation loss     < L2

    GSNA-8x
    total capacity      800M
    active parameters   ~25M
    validation loss     < L4

Depois, repetir com residency limitada:

    resident bytes      aproximadamente constantes
    bytes/token         aproximadamente limitados
    quality             continua melhorando com capacidade total

Se essa curva existir, a GSNA possui evidência de que a capacidade externa está realmente sendo utilizada.

# 30. Princípio fundador revisado

A genealogia completa passa a ser:

### Eyle

Como possuir memória crescente sem colocar toda memória no contexto?

**Resposta:** memória endereçável e materialização seletiva.

### MiniMaxBrain

Como possuir mais parâmetros do que cabem simultaneamente na RAM?

**Resposta:** parâmetros endereçáveis, pagináveis, verificáveis e reutilizáveis.

### Limite encontrado

O que acontece quando o runtime já foi extensivamente otimizado, mas o próprio modelo continua produzindo um padrão de acesso fisicamente ruim?

**Resposta:** otimização externa deixa de ser suficiente.

### GSNA

E se a própria rede for construída e treinada para que estado, capacidade, memória e compute sejam endereçáveis?

**Resposta proposta:** Resident Neural Core + Fast State Fabric + Capability Graph + Memory Graph + routing treinável + runtime paginado.

A versão 0.3 acrescenta uma correção importante:

**desacoplar capacidade externa não exige minimizar artificialmente a inteligência residente.**

O princípio não é:

\|C\| ≈ constante

O princípio é:

\|C\| ⟂arch \|Gₚ\|

ou, em linguagem direta:

**o RNC e o Capability Graph podem escalar em ritmos diferentes.**

Isso cria três eixos independentes de projeto:

\[ ResidentIntelligence \]

\[ AddressableCapability \]

\[ ActiveComputation \]

A GSNA existe para descobrir como combinar esses três eixos sob hardware real.

# 31. Hipótese final da versão 0.3

A hipótese científica da GSNA 0.3 é:

**Uma arquitetura neural pode manter um Resident Neural Core cuja escala é escolhida pelo melhor compromisso entre qualidade e custo, enquanto estados neurais percorrem dinamicamente um Capability Graph potencialmente muito maior. Se roteamento, estado, localidade, compute e residência forem aprendidos em conjunto, a capacidade total do sistema pode crescer mais rapidamente que seu working set físico e seu custo ativo por token.**

A primeira configuração de referência é:

\[ RNCWeightBytes=128 \]

Mas a arquitetura não depende desse número.

A pergunta de scaling passa a possuir duas dimensões:

Quality = f(\|C\|, \|Gₚ\|, ActiveCompute, B)

onde:

- (\|C\|) é o tamanho do RNC;

- (\|G_P\|) é o tamanho total do Capability Graph;

- ActiveCompute é a quantidade de capacidade realmente executada;

- \(B\) representa o orçamento físico.

A propriedade que buscamos não é “núcleo constante”.

É:

d\|C\| / d\|Gₚ\| não é imposto pela topologia

não ser imposto pela topologia.

Ou seja, aumentar o graph não deve obrigar aumento proporcional do core, embora experimentos possam mostrar que certos tamanhos de core aproveitam melhor certos tamanhos de graph.

A mudança de perspectiva é:

    não:
    modelo gigante + otimização para caber

    nem:
    núcleo mínimo + tudo externo

    mas:
    inteligência residente dimensionada pelo hardware
    +
    capacidade externa endereçável
    +
    compute ativo seletivo

O Eyle mostrou como não carregar toda a memória.

O MiniMaxBrain mostrou como não carregar todos os parâmetros.

A GSNA precisa provar a etapa que ainda não existe:

**não executar uma topologia inteira quando somente uma trajetória especializada é necessária — e fazer isso sem perder a capacidade de utilizar um espaço de conhecimento muito maior que o working set ativo.**

# 32. Próximo documento

O próximo documento recomendado é:

    GSNA 0.3 — RNC-128MB Minimal Graph Language Model Specification

Ele deve transformar D-002..D-016 em decisões concretas de implementação.

Deve conter no mínimo:

1.  tokenizer e vocabulário;

2.  accounting exato dos 128 MB;

3.  precisão e número de parâmetros;

4.  arquitetura do RNC;

5.  arquitetura do Fast State Fabric;

6.  State Packet tensor shapes;

7.  Capability Node tensor shapes;

8.  router forward pass;

9.  estratégia de gradiente para routing;

10. topology representation;

11. frontier scheduler;

12. integration/commit;

13. LM head;

14. causal training loop;

15. losses e seus schedules;

16. telemetry schema;

17. checkpoints;

18. testes de causalidade;

19. baseline RNC-only;

20. baseline Dense/MoE/SSM;

21. experimento Capability Scaling 1x/2x/4x/8x;

22. experimento Core Scaling 64/128/256/512 MB;

23. matriz 2D Core × Graph;

24. acceptance gates para avançar para paging físico.

Esse documento deve ser suficientemente concreto para que dois engenheiros independentes implementem o mesmo modelo sem precisar reinterpretar o whitepaper.

# Apêndice A — O que é herdado e o que é novo

| **Componente**                           | **Origem principal** | **Situação na GSNA 0.3**                                             |
|------------------------------------------|----------------------|----------------------------------------------------------------------|
| Memory Graph                             | Eyle                 | Herdado conceitualmente e reutilizável                               |
| Bounded context materialization          | Eyle                 | Herdado                                                              |
| Capability registry/contracts            | Eyle                 | Adaptado para nodes neurais                                          |
| Separation semantic/mechanical authority | Eyle                 | Herdado como invariante                                              |
| Parameter paging                         | MiniMaxBrain         | Herdado                                                              |
| Cache / residency                        | MiniMaxBrain         | Herdado                                                              |
| Leases / pin during compute              | MiniMaxBrain         | Herdado                                                              |
| Integrity / fail-closed                  | MiniMaxBrain         | Herdado                                                              |
| Native physical telemetry                | MiniMaxBrain         | Herdado                                                              |
| Dynamic compute graph                    | GSNA v0.1            | Núcleo da arquitetura                                                |
| Neural Microcore                         | GSNA v0.2            | Termo histórico; conceito corrigido na v0.3                          |
| Resident Neural Core (RNC)               | GSNA v0.3            | Terminologia canônica; núcleo residente, dimensionável e não-hub     |
| RNC-128MB baseline                       | GSNA v0.3            | Primeiro ponto experimental de RNCWeightBytes, não teto arquitetural |
| Core Scaling × Capability Scaling        | GSNA v0.3            | Eixos independentes de validação                                     |
| Budget-constrained training              | GSNA v0.2            | Mantido                                                              |
| Sparse training state                    | GSNA v0.2            | Mantido, fase posterior                                              |
| Fast State Fabric                        | GSNA v0.3            | Novo componente explícito                                            |
| State Packet                             | GSNA v0.3            | Novo contrato lógico                                                 |
| Direct capability edges                  | GSNA v0.3            | Formalizado                                                          |
| Frontier scheduler                       | GSNA v0.3            | Novo modelo de execução candidato                                    |
| Temporal commit contract                 | GSNA v0.3            | Nova decisão arquitetural                                            |

# Apêndice B — Fontes de implementação analisadas

## Eyle 2.7.5 Rev4.2.3

Referências principais do pacote:

    docs/memory-kernel.md
    docs/capability-providers.md
    docs/nucleus-cognitive-protocol.md
    eyle/runtime/context_materializer.py
    eyle/core/context_projection.py
    eyle/runtime/memory_graph.py
    eyle/capabilities/registry.py

Essas fontes sustentam as afirmações sobre:

- Memory Graph persistente;

- ausência de projeção automática global;

- budgets de materialização;

- capability contracts;

- registry mecânico;

- separação de autoridade semântica e física.

## MiniMaxBrain 0.4.1

Referências principais do pacote:

    docs/runtime-architecture.md
    docs/0.4_IMPLEMENTATION_STATUS.md
    docs/0.4.1_IMPLEMENTATION_STATUS.md
    MINIMAXBRAIN_0.4_SDD_COMPUTE_BOUND_2TPS_SPEC.md
    MINIMAXBRAIN_0.4.1_MEASUREMENT_INTEGRITY_SPEC.md
    native/src/mmb_pager.cpp
    native/src/mmb_runtime.cpp
    native/src/mmb_kernel.cpp
    benchmark-a2.json
    a2-diagnose.json

Essas fontes sustentam as afirmações sobre:

- router authority;

- paged expert parameters;

- placeholders;

- leases;

- cache;

- atomic paging behavior;

- native instrumentation;

- benchmark e paging bottleneck.

## GSNA

    Graph-State Neural Architecture — White Paper v0.1
    Graph-State Neural Architecture — White Paper v0.2

A v0.3 preserva a hipótese original da v0.1, incorpora os avanços físicos da v0.2, corrige a interpretação do núcleo como hub/minicore obrigatório e introduz o RNC dimensionável, o Fast State Fabric e os experimentos separados de Core Scaling e Capability Scaling.

# Conclusão

A GSNA não nasceu de uma tentativa abstrata de desenhar uma nova rede.

Ela surgiu de um caminho de engenharia.

Primeiro, o Eyle mostrou que memória crescente não precisa significar contexto crescente.

Depois, o MiniMaxBrain mostrou que capacidade paramétrica crescente não precisa significar residência integral.

A etapa seguinte surgiu quando a otimização chegou ao limite estrutural: o runtime ainda estava tentando adaptar uma arquitetura que não foi criada para esse ambiente.

A GSNA é a tentativa de inverter o problema.

Em vez de perguntar:

**Como fazemos um modelo gigante existente caber e rodar melhor?**

a pergunta passa a ser:

**Como construímos um modelo cuja própria organização neural já assume que somente uma pequena parte da capacidade estará ativa, residente e em trânsito a cada momento?**

A resposta ainda não está demonstrada.

Mas, após Eyle e MiniMaxBrain, grande parte da infraestrutura necessária já possui precedentes concretos.

O que resta é precisamente o núcleo científico da GSNA:

- Fast State Fabric;

- State Packet;

- routing aprendível;

- direct graph transitions;

- temporal commit;

- frontier scheduling;

- capacity scaling sob active compute limitado;

- Core Scaling sob budget residente explícito;

- interação 2D entre tamanho do RNC e tamanho do Capability Graph.

É isso que a próxima implementação precisa provar.
