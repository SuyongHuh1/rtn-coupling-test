# ccnn RTL — DE10-Nano (Cyclone V 5CSEBA6U23I7) / Quartus

Δ-입력 int8 인과 CNN(학습 완료, `results/r3/ccnn_diff_params.pkl`)의 스트리밍 Verilog 구현.
`rtnsim/quant.py::run_int`(numpy 정수 레퍼런스) 및 HLS C++과 **비트 단위로 동일**.

## 파일
| 파일 | 내용 |
|---|---|
| `ccnn_core.v` | 스트리밍 엔진 (샘플 1개 입력 → 32채널 S1=Σh, S2=Σh² 누적) |
| `ccnn_params.vh` | `Q_SHIFT` (=24) |
| `ccnn_w.hex` | 가중치 704 워드 × 256 bit (주소 = rombase(l) + o·K + k, 바이트 c = W[k][c][o]) |
| `ccnn_b.hex`, `ccnn_m.hex`, `ccnn_ms.hex` | bias / requant 배율 / skip 배율 (192 × 32 bit, 주소 l·32+o) |
| `tv_in.hex`, `tv_exp.hex` | 테스트 벡터: 8 trace × 2000 샘플 int8 입력, 정답 S1/S2 |
| `tb_core.v` | iverilog 테스트벤치 |
| `de10_selftest.v` | 보드 자가검증 top |
| `create_project.tcl` | Quartus 프로젝트 생성 (디바이스, 핀, SDC) |

## 구조
- L0: 7탭 입력 conv (7개 최근 입력을 한 벡터로 묶어 K=1로 처리)
- L1–L4: K=5, dilation 1/2/4/8 residual conv — 라인버퍼는 하나의 64×256bit RAM (M10K)
- L5: 1×1 conv → S1/S2 누적 (48 bit)
- 엔진: 출력채널 o, 탭 k마다 1사이클에 32-way int8 내적 (곱셈기 32개 + 파이프라인 adder tree),
  채널 종료 시 ReLU → `(acc·m + 2^23) >> 24` → [0,127] 포화, residual은 skip 재양자화 후 포화 덧셈
- **798 사이클/샘플** → 50 MHz에서 62.7 kS/s, PLL 100 MHz면 125 kS/s

## 검증 (클라우드에서 수행)
- iverilog: 8 trace × 2000 샘플(16,000 샘플) 전체 S1/S2 비트 일치 (PASS)
- 자가검증 top 시뮬레이션: PASS LED 점등, 정답 1개를 일부러 바꾸면 FAIL LED 점등 확인
- verilator lint: 오류 없음 (경고는 의도된 폭 확장/임시변수)
- yosys `synth_intel_alm -family cyclonev` 추정: ALUT ~3.5k, FF ~1.7k, M10K ~30(+테스트벡터),
  9×9 곱셈기 34 + 18×18 4 + 27×27 2 → DE10-Nano 자원의 10% 미만. Quartus 실제 수치는 다를 수 있음.

## Quartus에서 실행
1. 이 폴더 전체를 복사 (hex 파일이 `.v`와 같은 폴더에 있어야 `$readmemh`가 찾음)
2. Quartus → Tools → Tcl Scripts → `create_project.tcl` 실행
   (또는 명령행 `quartus_sh -t create_project.tcl`)
3. `ccnn_de10.qpf` 열기 → Processing → Start Compilation
4. Programmer → `output_files/ccnn_de10.sof` → JTAG로 5CSEBA6 장치에 다운로드
   (DE10-Nano는 체인에 HPS SOCVHPS + 5CSEBA6이 보임 — FPGA 쪽에 .sof 지정)
5. LED 확인: LED0 하트비트, LED2 완료, **LED3 = PASS**, LED4 = FAIL, LED7–5 = 처리 중인 trace 번호(끝나면 7).
   KEY0 누르면 다시 실행.
6. 속도 측정: Signal Tap으로 `sample_period`(샘플 간 사이클, 기대값 798), `run_cycles`, `ref_cycles`(50 MHz 기준 실제 시간)를 읽음.
   PLL 클럭으로 돌리려면 IP `pll_sys`(PLL Intel FPGA IP, outclk0) 생성 후 매크로 `USE_PLL=1` 지정.

## 핀 (Terasic DE10-Nano 매뉴얼 기준, 3.3-V LVTTL)
FPGA_CLK1_50 = V11, KEY0 = AH17, KEY1 = AH16,
LED0–7 = W15, AA24, V16, V15, AF26, AE26, Y16, AA23.
보드 리비전에 따라 다를 수 있으니 컴파일 전에 System CD의 핀 표와 대조하세요.

## 다음 단계
- HPS(ARM)와 연결: Avalon-MM 슬레이브로 감싸 Linux에서 샘플 입력/S1·S2 읽기, 헤드 MLP는 ARM에서 float
- 실시간 입력: 보드 LTC2308 ADC → Δ 및 정규화(접두 512샘플 표준편차) 블록 추가
