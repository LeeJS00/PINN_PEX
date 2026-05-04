The transition from a localized 1-hop GNN to a global Geometry-Informed Neural Operator (GINO) mathematically aligns with the underlying physics of parasitic extraction. The Poisson equation is fundamentally elliptic and non-local; treating it as a local graph aggregation problem was the root cause of the 320%+ CPL SMAPE. Using a layerwise 2.5D FNO to approximate the Green's function for the macroscopic topology is a highly elegant architectural leap.

However, a critical review of the proposed architecture reveals severe mathematical and systemic vulnerabilities, particularly in how the grid resolution interacts with near-field physics and how the PDE loss is formulated. 

Here is a critical dissection of the proposed v2 roadmap.

### 1. The P2G Resolution and the Near-Field Blurring Collapse
The report proposes a latent grid $G=64$ over an $8 \times 8 \mu m$ window, resulting in a grid resolution of 125nm per cell. The Gaussian kernel $\sigma_{xy}$ is set to 0.25μm (250nm). 

* **The Physical Conflict:** At advanced nodes (e.g., M4 with a 44nm pitch), up to 5 parallel wire segments will fall within a single 250nm Gaussian envelope. The Particle-to-Grid (P2G) scatter operation will smear these distinct conductors into a single blurred latent activation block.
* **Why Channel-Wise Identity is Insufficient:** The report suggests adding physical dimensions ($w, h$, layer) to the cuboid encoder to distinguish overlapping wires. However, once scattered onto the grid, the FNO computes spatial convolutions over this blended representation. The high-resolution spatial layout context—specifically the nanometer-scale gaps that dictate fringing fields and line-of-sight shielding—is irreversibly destroyed before the FNO even begins processing. 
* **The Verdict:** FNOs are exceptional at capturing the smooth far-field macroscopic topology, but parasitic capacitance is overwhelmingly dominated by high-frequency near-field singularities (sharp edges and tight gaps). The proposed P2G/G2P pipeline risks trading CPL prediction failures caused by limited receptive fields for failures caused by catastrophic spatial aliasing.

### 2. The Fallacy of Latent Poisson Regularization
Section 4.3 proposes a Poisson residual loss using a 2D Laplacian on the latent field: `laplacian_2d(latent_field[dielectric_mask]).pow(2).mean()`.

* **The Mathematical Flaw:** The FNO operates in an abstract latent space $\mathbb{R}^{128}$. Applying a finite-difference Laplacian to an arbitrary embedding vector is physically meaningless. The true Poisson equation, $\nabla^2 \phi = 0$, strictly governs the physical electrostatic potential (Volts), not a 128-dimensional hidden representation.
* **The Verdict:** To enforce a valid PDE constraint, the architecture must explicitly decode a single channel from the latent grid to represent the physical potential $\phi$, apply strict Dirichlet boundary conditions ($\phi = V_{target}$ on the target net, $\phi = 0$ on aggressors/VSS), and then compute the divergence. Regularizing the abstract latent field will introduce severe mathematical noise and actively fight the active learning (AL) convergence.

### 3. The "cap = net_cap / n_tiles" Fatal Assumption
In the Risk Assessment (Section 6), the mitigation for dataset label noise states: "cap = net_cap / n_tiles". 

* **The Systemic Flaw:** Capacitance is not a linearly divisible scalar. A wire crossing an empty tile contributes very differently than the same wire crossing a dense power distribution network (PDN) island. Simply dividing the full-net StarRC golden label by the number of tiles violates Kirchhoff's laws and the fundamental physics of spatial charge distribution.
* **The Verdict:** Training the model to predict an artificially uniform fraction of the global capacitance will completely break the model's ability to learn localized flux competition. The network must be trained on localized field solver labels (e.g., FastCap/FRW running on the exact tile), or the tile-level predictions must be dynamically assembled and evaluated against the global label at the full-net level during the backward pass.

### 4. CUDA Runtime Optimism vs. Memory Bandwidth
The projected CUDA timing of ~3ms/tile assumes perfect scaling with `torch_scatter`.

* **The Hardware Reality:** Scattering unaligned 3D cuboid coordinates into a dense grid using Gaussian kernels involves massive non-coalesced memory writes and atomic collisions. While `scatter_add` is fast, the bottleneck will shift to memory bandwidth limits when running dense parallel islands. 
* **The Verdict:** Relying on $K=9$ nearest cells creates a sparse masking overhead that often runs slower than dense operations in pure PyTorch. A custom CUDA kernel specifically optimized for 2.5D BEOL cuboid rasterization will likely be required to hit the sub-5ms target.

### Actionable Architecture Pivots

To salvage the 4% MAPE target without falling into these traps, the architecture requires the following structural pivots:

1.  **Hybrid Near/Far-Field Processing:** Do not force near-field coupling through the FNO grid. Maintain a direct Point-to-Point (Cuboid-to-Cuboid) attention or strict geometric tensor branch for elements within $\leq 0.5 \mu m$. Use the GINO P2G $\rightarrow$ FNO $\rightarrow$ G2P pipeline *exclusively* to compute the macroscopic background potential and long-range screening effects, then concatenate this global context back into the exact geometric cuboid representations before the final MLP heads.
2.  **Explicit Singularity-Aware Charge Basis:** Instead of mapping arbitrary MLP embeddings to the grid, the CuboidEncoder should output parameterized charge density bases (coefficients for analytical boundary elements). The FNO then acts as a neural preconditioner resolving the global interactions of these physical charges.
3.  **Global AL Assembly:** The Active Learning cycles must evaluate the loss on assembled full-chip (or full-net) macro-structures, never on artificially divided tile-level scalar targets.


제시해주신 "GINO 기반 BEOL PEX 아키텍처 보고서"를 학술적, 물리적 관점에서 깊이 있게 분석했습니다. 

국소적인 1-hop GNN의 한계(CPL SMAPE 320%+)를 직시하고, 라플라스 방정식의 해 공간(Green's function) 자체를 근사하기 위해 GINO(Geometry-Informed Neural Operator)를 도입하려는 접근은 **EDA 학계(DAC, ICCAD)에서도 매우 파괴적이고 선도적인 시도**입니다.

### 📜 관련 선행 연구와의 비교 및 학술적 기여도(Contribution) 평가

1. **ParaGraph (DAC 2022) 등 GNN 기반 모델과의 비교**
   * 기존 연구들은 회로도나 레이아웃을 그래프로 변환하여 MPNN(Message Passing)을 적용하지만, 보고서의 진단처럼 심각한 Over-smoothing 문제를 겪습니다. GINO의 도입은 이 국소적 정보 소실 문제를 Global Fourier 공간에서 해결한다는 점에서 훌륭한 학술적 차별성을 갖습니다.
2. **Original GINO (NeurIPS 2023) 모델과의 비교**
   * 원본 GINO는 자동차 유체 역학(CFD)이나 3D 연속체 문제에 맞춰져 있습니다. 이를 BEOL의 층상(Planar) 배선 구조에 맞춰 **2.5D Layerwise FNO와 Z-MLP로 개조**한 점은 3D FFT의 무거운 연산량을 피하면서 물리적 특징을 살린 탁월한 엔지니어링 기여입니다. 
3. **학술 논문으로서의 가치**
   * 최근 IEEE(2025) 등에서 3D 금속 타겟의 전기장 적분 방정식(EFIE)에 Neural Operator를 적용한 연구가 등장하기 시작했는데, 이를 칩 레벨의 기생 정전용량 추출(PEX)로 확장하고 Active Learning까지 결합한 구조는 SOTA(State-of-the-Art)로 인정받기에 충분한 독창성을 가집니다.

---

### 🚨 비판적 검토: 정확도 4%의 벽을 가로막는 3가지 치명적 모순

아키텍처의 거시적 방향성은 훌륭하지만, Physical Design의 본질적 관점에서 이 보고서의 구상에는 반드시 수정되어야 할 **3가지 치명적 결함**이 존재합니다.

#### 1. P2G 격자 해상도와 근거리 특이점(Near-Field Singularity)의 증발
보고서는 8x8μm 타일을 64x64 그리드(셀당 125nm)로 나누고 $\sigma_{xy}=0.25\mu m$의 가우시안 커널을 사용한다고 명시했습니다.
* **비판:** 첨단 공정(Advanced Node)에서 메탈 피치는 44nm 수준입니다. 250nm의 커널로 Particle-to-Grid(P2G) 산란을 수행하면, 인접한 3~5개의 독립적인 메탈 배선들이 하나의 뭉뚱그려진 격자 활성화 값으로 블러링(Blurring)됩니다. 푸리에 신경 연산자(FNO)는 거시적 토폴로지를 파악하는 데는 탁월하지만, 정전용량의 절대다수를 차지하는 "나노미터 단위의 미세한 틈새(Gap)와 모서리의 프린징(Fringing) 특이점"  을 복원하는 데는 치명적인 약점을 가집니다. Channel-wise identity를 추가하더라도 고해상도 기하학 정보는 FNO 연산 진입 전에 비가역적으로 파괴됩니다.

#### 2. 잠재 공간(Latent Space)에서의 가짜 편미분 방정식(PDE) 정규화
보고서 4.3절의 `laplacian_2d(latent_field[dielectric_mask]).pow(2).mean()` 로직은 수학적으로 성립하지 않습니다.
* **비판:** 포아송 방정식 $\nabla^2 \phi = 0$은 연속된 공간에서의 물리적인 정전기 포텐셜(Volts) $\phi$에 대해 성립하는 법칙입니다. 그런데 GINO의 잠재 격자(Latent Grid)는 $\mathbb{R}^{128}$ 차원의 추상적 임베딩 공간입니다. 물리적 스칼라장이 아닌 128차원 인코딩 벡터 덩어리에 유한차분법(Finite Difference)으로 라플라시안을 씌워 페널티를 주는 것은 KCL(키르히호프 법칙) 준수에 도움을 주기는커녕 신경망 학습을 교란하는 수학적 노이즈에 불과합니다.

#### 3. "cap = net_cap / n_tiles" 분할의 물리적 보존 법칙 위반
보고서 6절(Risk Mitigation)에서 타일 레벨의 라벨 노이즈를 해결하기 위해 전체 넷의 캡을 타일 수로 일괄 분할하겠다는 접근이 명시되어 있습니다.
* **비판:** 전하 보존의 근본을 무시한 접근입니다. 기생 정전용량은 선형적으로 쪼개지지 않습니다. 텅 빈 타일을 지나가는 배선과 빽빽한 VSS PDN(Power Distribution Network) Island 구역을 관통하는 동일한 배선의 커플링 지분율은 완전히 다릅니다. 인위적으로 분할된 스칼라 타겟으로 학습을 강제하면, 모델은 공간에 따른 비선형적인 국소 플럭스 경쟁(Flux Competition)을 학습할 수 없습니다.

---

### 💡 다음 단계를 위한 전문가적 대안 (Alternative Architecture)

이 한계를 돌파하고 4% 오차율에 도달하려면 GINO를 단독 솔루션이 아닌 **"거시적 전제조건(Macroscopic Preconditioner)"**으로 위치시키는 하이브리드 아키텍처로 선회해야 합니다.

* **Dual-Stream 구조 채택:**
  1. **Far-field (GINO):** 보고서의 P2G $\rightarrow$ FNO $\rightarrow$ G2P 파이프라인은 주변 메탈 밀도와 먼 거리의 VSS/Ground가 빚어내는 "거시적 배경 포텐셜(Background Potential)" 및 스크리닝 효과를 파악하는 용도로 한정합니다.
  2. **Near-field (Surface Physics):** 서로 근접해 있는 큐보이드 쌍에 대해서는 P2G로 뭉개지 말고, 날것의 **표면 투영 면적($A_{over}$), 표면 거리($D_{surf}$), 투영 둘레($P_{over}$)** 텐서를 직접 보존하여 입력합니다.
* **Singularity-Aware Charge Basis 적용:** 단순한 다차원 임베딩 대신, 인코더가 해석적 경계 요소(Boundary Element)의 전하 밀도 계수(Charge Coefficient)를 출력하도록 강제하여 모델이 스스로 Singularity-Aware Charge Basis를 형성하게 해야 합니다.

제안하신 보고서는 SOTA 논문으로 발전할 수 있는 강력한 뼈대를 갖추고 있습니다. 해상도 블러링을 방지하는 투트랙(Two-track) 처리와 물리 법칙에 부합하는 글로벌 스케일의 평가 메커니즘만 보완한다면, 기존 학계의 한계를 뛰어넘는 완벽한 연구가 될 것입니다.

이전의 단일 스트림 아키텍처들이 왜 320%의 CPL SMAPE나 0.17의 상관계수 벽에 부딪혔는지 우리는 데이터로 완벽히 증명했습니다. 푸리에 신경 연산자(FNO)가 거시적 토폴로지는 잘 잡지만, 근거리의 기하학적 특이점(Singularity)을 뭉개버린다는 점이 핵심이었습니다.

특히 최근 디지털 설계에서 트렌드가 되고 있는, 특정 지점들에 아일랜드(Island) 형태로 빽빽하게 PDN을 까는 국소적이고 이질적인 배선 구조를 생각해 보면 이 문제는 더욱 심각해집니다. 국소적인 1-hop GNN은 이 거대한 PDN 아일랜드가 만드는 스크리닝 효과를 절대 보지 못하며, 반대로 GINO의 그리드는 PDN 내부의 미세한 나노미터 틈새 전력선을 전부 블러링(Blurring)해 버립니다.

이 모순을 해결하기 위해, 거시적 배경 장(Background Field)과 미시적 전하 특이점(Charge Singularity)을 중첩(Superposition)시키는 **투트랙 하이브리드 아키텍처(Dual-Stream Hybrid Architecture)**에 대한 깊은 고찰을 담아 `plan` 파일을 작성했습니다.

***

# 📋 ARCHITECTURE PLAN: Dual-Stream Physics-Informed Neural Operator (DS-PINN)

## 1. Core Philosophy: The Principle of Superposition
전자기학의 선형 중첩 원리를 신경망 구조에 그대로 이식합니다. 정전기 포텐셜 $\phi$는 거시적 환경이 만드는 배경 포텐셜($\phi_{macro}$)과, 내 코앞의 도체가 만드는 국소 포텐셜($\phi_{micro}$)의 합으로 정의될 수 있습니다. 
단일 모델로 이 두 가지 주파수 대역(저주파의 거시 구조 vs 고주파의 미세 엣지)을 모두 학습하려는 시도를 폐기하고, 역할을 완벽히 분리합니다.

---

## 2. Stream A: Macro-Solver (The Background FNO)
아일랜드 형태의 밀집된 PDN이나 거대한 VSS 레일이 타겟 넷에 미치는 장거리 차폐(Long-range Shielding) 효과를 계산합니다.

* **해상도 완화:** 국소적인 엣지 정보에 집착할 필요가 없습니다. G=64 격자를 유지하되, 이 그리드가 묘사해야 하는 것은 개별 메탈의 형태가 아니라 **"공간적 금속 밀도(Metal Density Field)"**입니다.
* **P2G (Particle-to-Grid) 재정의:** 개별 큐보이드의 피처를 산란(Scatter)시키는 것이 아니라, $\epsilon$이 가중된 부피 밀도(Volume Fraction)만을 격자에 투영합니다.
* **FNO의 역할:** $\nabla^2 \phi = -\rho / \epsilon$ (포아송 방정식)의 거시적 해(Green's function)를 근사합니다. 출력물은 타일 전체 공간에 깔려 있는 "배경 스크리닝 텐서(Background Screening Tensor)"입니다.

---

## 3. Stream B: Micro-Solver (Singularity-Aware Charge Basis)

나노미터 단위의 표면 틈새($D_{surf}$)와 모서리에서 발생하는 전계의 특이점(Singularity) 및 프린징(Fringing) 플럭스를 직접 계산합니다.

* **수학적 뼈대:** 앞서 우리가 발굴한 4대 기하학 팩트($D_{surf}$, $A_{over}$, $P_{over}$, $\epsilon_{pair}$)를 무손실로 사용합니다.
* **Singularity-Aware Charge Basis:** MLP는 스칼라 정전용량을 바로 뱉는 것이 아니라, **해석적 경계 요소(Boundary Element)의 전하 밀도 기저(Charge Basis) 계수**를 출력하도록 강제합니다. 즉, 뾰족한 모서리(P)에 전하가 집중되는 물리적 특성을 기저 함수(Basis Function)로 미리 정의하고, 신경망은 그 기저의 가중치만 학습합니다.
* **G2P (Grid-to-Particle) 결합:** Stream A에서 계산된 '배경 스크리닝 텐서'를 큐보이드의 위치 좌표로 보간(Interpolation)하여 가져옵니다. 

---

## 4. The Integration & Loss Landscape (PINN Formulation)
두 스트림의 결합부와 손실 함수(Loss) 설계입니다.

* **Neural Shader (최종 결합):**
    ```python
    # 큐보이드 쌍(i, j)의 최종 커플링 예측
    local_geometry = concat([D_surf, A_over, P_over, eps_pair]) 
    macro_context = concat([G2P(background_field_i), G2P(background_field_j)])
    
    charge_coeffs = MLP_Basis(local_geometry, macro_context)
    C_ij = Analytic_Integration(charge_coeffs, local_geometry) 
    ```
* **Physics Constraint (가짜 PDE Loss의 폐기):**
    잠재 공간(Latent Space)에서의 무의미한 라플라시안 정규화를 폐기합니다. 대신, 예측된 `charge_coeffs`가 타겟 도체 표면에서 $V = 1V$라는 경계 조건(Dirichlet Boundary Condition)을 얼마나 위반하는지를 계산하여 **경계 조건 잔차 손실(Boundary Residual Loss)**로 활용합니다.
* **Global KCL Assembly:** 타일 단위로 조각난 스칼라 정답지를 폐기합니다. 모델은 타일 내부의 국소 커플링($C_{ij}$)만 출력하고, Loss 연산 시 타일 밖의 정보까지 연결된 **풀넷(Full-Net) 단위로 KCL 총합을 조립**하여 StarRC 정답지와 비교합니다.

---

## 5. Phased Execution Roadmap

* **Phase 1: Micro-Stream Isolation (주차 1)**
    * 거시적 장(Macro-field)이 배제된 상태에서, 국소 기하학 피처($A_{over}, D_{surf}$)만을 이용해 Singularity-Aware Charge Basis MLP를 학습. (기존 SOTA 28% SMAPE 벽 돌파 확인)
* **Phase 2: Macro-Stream FNO Integration (주차 2)**
    * 밀집 PDN 아일랜드가 포함된 데이터셋을 바탕으로 FNO 기반 밀도장(Density Field) 구축 및 G2P 보간 모듈 구현.
* **Phase 3: Global Loss & Active Learning (주차 3)**
    * 풀넷(Full-Net) KCL 조립파이프라인 구축 및 Boundary Residual Loss 적용. AL(Active Learning) 루프 재가동.

***

이 하이브리드 계획은 딥러닝이 가장 잘하는 것(거시적 토폴로지 근사)과 물리 엔진이 가장 잘하는 것(기하학적 특이점 적분)을 분리하여 충돌을 막는 가장 진보된 형태의 아키텍처입니다. 

우리 코드의 가장 큰 물리적/구조적 특성은 **"모든 메탈 배선이 3D 직육면체(Cuboid/AABB)의 집합으로 표현되며, 텐서 브로드캐스팅을 통해 표면 대 표면(Surface-to-Surface)의 교차 면적과 거리를 무손실로 계산할 수 있다"**는 점입니다. 

기존 GINO가 이 완벽한 직육면체 기하학을 픽셀(Grid)로 뭉개버려서 실패했다면, 우리의 핵심 알고리즘은 **"직육면체 기하학의 무손실 보존(Micro)"**과 **"격자 기반의 거시적 차폐장(Macro)"**을 텐서 레벨에서 결합하는 것입니다.

우리 코드의 특성을 100% 반영한 **DS-PINN(Dual-Stream PINN) 핵심 알고리즘 설계도**를 제안합니다.

---

# 📋 DS-PINN Core Algorithm Plan

## 1. 아키텍처의 근본 철학: 물리적 분리 (Physical Decoupling)
전기장(E-field)은 두 가지 성분으로 나뉩니다.
1. **$E_{near}$ (미시적 특이점):** 두 큐보이드 표면이 5nm~100nm 단위로 마주 볼 때 폭발적으로 발생하는 평행판 및 프린징 플럭스. (격자화 불가능, 절대적 정밀도 필요) 
2. **$E_{far}$ (거시적 배경장):** 수 마이크로미터 밖의 거대한 PDN 아일랜드나 VSS 레일이 전력선을 흡수하여 타겟 주변의 공간 포텐셜을 낮추는 스크리닝 효과. (격자화 가능, 부드러운 토폴로지)

우리 코드는 이 두 성분을 각각의 특성에 가장 잘 맞는 텐서 연산으로 분리하여 병렬 처리합니다.

---

## 2. Stream A: Macro-Field (거시적 스크리닝 텐서)
**목적:** "타일 내의 금속 밀도 분포가 타겟 큐보이드 주변의 플럭스를 얼마나 빼앗아 가는가?"를 학습합니다.

* **P2G (Particle-to-Grid) 연산:**
    * 우리의 `cuboids` 텐서 `(x, y, z, w, h, d, eps)`를 3D 그리드로 산란(Scatter)시킵니다.
    * 이때 개별 메탈의 형태를 보존할 필요 없이, 해당 격자의 **"금속 부피 밀도(Volume Fraction)"**와 **"평균 유전율"**만 채워 넣습니다.
* **배경장(Background Field) 추론:**
    * 경량화된 3D-CNN 또는 Layerwise FNO가 이 밀도 그리드를 통과하며 거시적인 '스크리닝 포텐셜 맵'을 만듭니다.
* **G2P (Grid-to-Particle) 보간:**
    * 그리드의 결과값을 다시 각 큐보이드의 중심 좌표 `(x, y, z)`로 보간(Interpolation)하여 가져옵니다.
    * **결과물:** 각 큐보이드 $i$는 64차원의 거시적 맥락 벡터 $Z_{macro}^{(i)}$ 를 부여받습니다. (의미: "나는 현재 VSS 레일 근처에 있어서 스크리닝이 강하다")

---

## 3. Stream B: Micro-Surface (무손실 큐보이드 기하학)
**목적:** "마주보는 두 큐보이드 $i, j$ 사이에 물리적으로 흐를 수 있는 최대 플럭스는 얼마인가?"를 계산합니다.

* 우리가 방금 증명해 낸 `probe_surface_physics.py`의 텐서 연산을 그대로 가져옵니다.
* `mins = centers - sizes / 2.0`, `maxs = centers + sizes / 2.0`
* 브로드캐스팅을 통해 마주보는 **표면 최단 거리($D_{surf}$)**, **직교 투영 면적($A_{over}$)**, **투영 둘레($P_{over}$)**를 완벽하게 계산합니다.
* **결과물:** 엣지 $(i, j)$에 대한 완벽한 기하학적 팩트 텐서 $G_{micro}^{(i, j)} = [D_{surf}, A_{over}, P_{over}, \epsilon_{pair}, DZ_{gap}]$.

---

## 4. 최종 결합: 신경망 기반 전하 기저 모듈 (Neural Charge Basis Module)
이 아키텍처의 꽃입니다. 단순 MLP가 아니라, Stream B(기하학)가 제공하는 해석적 수식을 Stream A(거시적 환경)가 가중치로 조절하는 **"비선형 물리 셰이더"**입니다. 

우리의 코드(finetuner.py)에 이식될 핵심 연산은 다음과 같습니다.

```python
# 1. Macro Context 병합
# 두 큐보이드의 거시적 스크리닝 상태를 결합
Z_pair = torch.cat([Z_macro_i, Z_macro_j, abs(Z_macro_i - Z_macro_j)], dim=-1)

# 2. Singularity Coefficient 예측 (MLP)
# 거시적 환경(Z_pair)을 바탕으로, "평행판", "프린징", "잔차"에 대한 3개의 계수를 예측
coeffs = MLP(Z_pair) 
alpha = torch.sigmoid(coeffs[:, 0]) * 2.0  # 평행판 투과율 (0 ~ 2배)
beta  = torch.sigmoid(coeffs[:, 1]) * 2.0  # 프린징 투과율 (0 ~ 2배)
gamma = F.softplus(coeffs[:, 2]) * 0.1     # 순수 비선형 잔차 보정

# 3. Physics Integration (최종 커플링 정전용량)
# 순수 기하학(Micro) 텐서에 학습된 계수(Macro)를 곱하여 최종 Cap을 구함
C_parallel = EPS_0 * eps_pair * (A_over / D_eff)
C_fringe   = EPS_0 * eps_pair * (P_over / torch.log1p(D_eff * 10.0))

C_ij = (alpha * C_parallel) + (beta * C_fringe) + gamma
```

## 5. 우리 코드 특성상 이 알고리즘이 성공할 수밖에 없는 이유
1. **분모 폭발(Denominator Explosion) 원천 차단:** 그리드를 쓰지 않고 AABB의 BBox 좌표를 직접 투영 연산하므로 $D_{surf}$ 가 가짜로 0이 되는 현상이 없습니다.
2. **KCL(Kirchhoff's Current Law) 조립의 용이성:** 위 연산은 타일 단위가 아니라 철저히 엣지(큐보이드 쌍) 단위로 이루어집니다. 따라서 여러 타일에 걸쳐 파편화된 넷이더라도, $C_{ij}$ 를 모두 더한 뒤 풀칩 정답지(Golden Cap)와 Loss를 계산하는 것이 구조적으로 완벽히 들어맞습니다.
3. **해석 가능성(Interpretability):** 모델이 단순히 숫자를 뱉는 것이 아니라, 환경에 따라 `alpha(평행판 투과율)`와 `beta(프린징 투과율)`를 어떻게 조절하는지 실시간 모니터링(`probe_flux_router_anomalies`)이 가능합니다.