```mermaid
flowchart TD

    %% Novice Stage
    subgraph Novice Stage
        N1[Background Node]
        N2[Novice Quiz Node]
    end

    %% Intermediate Stage
    subgraph Intermediate Stage
        I1[Intermediate Quiz Node]
        I2[Intermediate Reflection Node]
    end

    %% Expert Stage
    subgraph Expert Stage
        E1[Expert Quiz Node]
        E2[Expert Reflection Node]
    end

    %% Edges - Novice Stage (병렬)
    N1 -->|독립 실행| END
    N2 -->|독립 실행| END

    %% Edges - Intermediate Stage (직렬)
    I1 --> I2 --> END

    %% Edges - Expert Stage (직렬)
    E1 --> E2 --> END
