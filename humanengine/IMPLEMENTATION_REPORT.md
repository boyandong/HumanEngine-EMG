# HumanEngine-EMG v0.1 科学内核实施报告

> 历史实施记录。独立审计发现 F01–F08 缺陷／provenance 差异，旧 59 项通过不能作为放行依据。
> 当前修复与验收以 REMEDIATION_REPORT.md、REMEDIATION_EVIDENCE.json 和 SESSION_HANDOVER.md 为准。
> 此处旧冻结完整性声明不能当作当前状态；原 FROZEN_BASELINE.json 未重写。

日期：2026-09-19。交付状态：**科学内核已实现，本地合成 CPU 正确性验收通过；尚无真实老师或 HE 科学训练结果。**

全部新文件位于 `C:\Users\董伯言\Desktop\fingers\emg2pose_handstate\humanengine`。
没有修改已有科学流水线、设计原稿、根 AGENTS 或冻结评价代码。实现不是改进姿态精度实验。

## A. SPEC RECONCILIATION

读取并依据：项目根 AGENTS、FOCUSED_REVIEW_MULTI_API_RESIDUAL、FOCUSED_REVIEW_HANDOVER、
HUMANENGINE_EMG_V0_1_ARCHITECTURE、原 SESSION_HANDOVER，以及用户本次实施任务。
任务原文另存为 SPEC_IMPLEMENTATION_TASK.md，并核对其与附件哈希一致。

增补只在多 API 家族预算/持续接入、残差目标、涉及 R 的 API 路由上覆盖原稿。
当前/未来 API 都读 `[S,detach(R)]`；未保留旧 future→R 直接反向路径。
RMS/band 已降为弱 observed 锚。原稿的前端结构、A/F/U/E 顺序、因果节拍、流式状态、
几何/VICReg 规则和 burn-in 仍适用。用户这次实施授权覆盖历史设计任务的“不实施”停止点，
但不授权真实训练或远端运行。

实际核对了 frozen baseline 的 data.py、utils.py、inference/predict_pose.py、canonical_hand.py
及官方 joint-name 解析接口；没有导入基线训练栈或改写其 loader。官方 IK valid 是
not-all-near-zero，finite 检查在新 lane 额外施加；不能把 NaN 或缺标签当有效零值。

非平凡工程选择登记在 IMPLEMENTATION_MAP.md：按流显式不可变状态、变长输出 valid mask、
周期 Hann、float64 时戳、固定目标尺度、有限求解失败的显式 fallback、非有限梯度的前置拒绝。
这些不是新增科学结论。CAGrad 数学已由增补给出；本轮未重开外部综述或替换算法。
独立老师健康阈值/真实训练预算必须后续预登记，未以猜测值替代真实实验配置。

## B. FILES CREATED

完整逐文件清单见附录 B1。主要入口为 api.py、contracts.py、registry.py、manifest.py，
以及 core/data/heads/objectives/teacher/optimization/continual/diagnostics/configs/tests。
另交付需求映射、用户任务原文、README、交接、JUnit 结果、冻结快照和实施证据 manifest。
pytest 缓存只写新 lane，属于可再生成测试产物，不是科学结果。

## C. FILES MODIFIED

**已有文件：NONE。** 本次创建的文件在实施中迭代完善；没有改动已有 root/baseline/
representation/inference/DB9/S3–S6 文件，也未初始化或重构 Git。

## D. CORE IMPLEMENTATION

- A 为输入 affine，U 为 F 后的 rank-8 残差适配；通用模式 identity/frozen。
  个体模式只开放授权 A/U，offset 没有噪声依据时保持冻结。适配器加载核对 core/statistics/
  input-schema 与版本，下一次 step 重置旧状态。
- F 为原稿的三层因果下采样卷积加两层 depthwise 残差块，逐时刻 channel LayerNorm。
  默认 anchors=39,79,…；卷积仅缓存有限左历史，不做未来插值或完整历史重算。
- E 在 A 之前读取 raw published EMG，因果 40 ms log-RMS，使用明确冻结统计。
- S256/R128 为独立 LSTM；S 读 U(F(A(x)))，R 再拼 E；H 是两个 hidden 的拼接。
- RuntimeState 单独保存双 h/c、卷积缓存与 phase、E 缓存、时戳、身份/adapter/reset 信息。
  H 不被称作完整 Markov/belief state。
- sequence 与任意 chunk step 数值一致；每条流独立重置。时戳断点/乱序、无效信号、
  recording/user/adapter/explicit-reset 处理明确，标签缺失不重置状态。
- training_sequence 每个视图独立 burn-in，显式 detach 后才进入计分段的 BPTT。

## E. API ROUTING

默认仅有当前/未来 pose API。APIHead 提供每家族私有 projector 和 head；通用输入为
`concat(S, R.detach())`。Geometry 独立 G(S)，不新增家族票数。
逐项 autograd 检查 API current/future→F/S/private，R 参数不获梯度。

shadow-gradient 仅诊断性地开放 R 路径，使用新缓存，保留参数/缓冲区、训练模式、RNG、
梯度和优化器状态。只读诊断不无故增加参数版本计数。测试确认 shadow 能观察被屏蔽的敏感性。
共享 F 的变化仍会影响 R 数值，未作“API 完全不能影响 residual”的声明。

## F. TEACHER IMPLEMENTATION

独立 causal frontend + 两层 128 LSTM；独立 student/EMA/predictors/M_T/统计和状态。
EMA 默认 .999/.001，只更新自该 student；不共享 HE 参数、buffer、初始化 checkpoint 或优化器。

三种上下文 .1/.5/2 秒均结束于 t，按窗左端 reset，保留两层末端 hidden，共六个目标。
目标支持检查 recording、finite、有效信号和连续时戳。没有当前 target 使用未来 EMG。

同一实际时间遮挡视图用于三个尺度；约 30% 支持被遮挡，各计分窗要求 20%–50%，包含末端块。
student 从自身输入重算所有路径；显式掩码验证不依赖“值是否变化”，因休息信号本来可以为零。

准备损失为 masked + .1 observed + .05 physical + .01 direct-hidden variance。
M_T 读取第二层 clean student hidden，跨尺度共享。前向本身不暗中更新 EMA 或统计。
提供 visible-context 参照、train-only statistics fit、健康检查和冻结导出接口。

导出要求固定预算、EMG-only development 健康证据及匹配的 checkpoint step；真实老师不能用
synthetic statistics。测试里的未训练老师仅为显式 synthetic wiring fixture。
部署 HumanEngine 不包含老师。

## G. RESIDUAL IMPLEMENTATION

P_R 为 128→128→768，共用于 masked/observed R，输出拆成三尺度×两层×128。
M_A 为独立物理锚头。严格计算：

`L_R = L_mask_lat + .1 L_obs_lat + .05 L_phys + .01 L_var_R`。

目标 stop-gradient，Huber delta=1；每个尺度/层独立按有效项均值，再等权聚合。
三种观察项分母独立，报告逐 target 有效数及误差。方差直接作用 R，默认需要至少64个时间分隔
anchors、8 recordings、4 users；不满足明确返回 NA，不伪造有效零值或重复邻帧支持。
没有加 residual whitening、orthogonality、subject adversary 或额外未来潜在目标。

物理目标采用原稿的40 ms log-RMS、100 ms周期Hann单边periodogram及五粗频带，包含单边能量
倍率和频率积分。正弦能量与幅值测试提供了独立数值参照；不把摘要称为力/疲劳/生理真值。

## H. MULTI-API TRAINING SYSTEM

APIRegistry 保存家族、版本、components、source IDs、mask 定义、冻结 reference/floor/beta、
private IDs、replay policy 和 route version。五个 pose 数据源不产生五份 API 预算。
当前/未来/geometry 的 beta 为1/.5/.05，future horizon 在其分量内部平均。

家族→稳定 acquisition source→user→session→recording→window 的分层采样消除 alias 重复。
还提供按分量有效目标数补齐的 quota plan 与截取 mask；无法补齐为 NA，禁止部分 optimizer update。
RNG、游标、窗口身份和 exposure 可保存/恢复。

reference scales 由 train-only 固定常量参照生成后冻结，提供回归/class prevalence/teacher-zero
baseline 计算接口；不是当前 batch loss、移动难度或梯度范数。
base budgets 为 API .8/K、reserve .2；reserve=(L_R/a_R+.1 VICReg)/1.1。
无 pose 标签的 EMG 可以计算 reserve，不生成 pose=0 的有效监督。

## I. CAGrad

使用完整 F/S/R 参数向量，未触达块填零；private heads/teacher/A/U 不纳入求解。
K=1 用 g0，K>=2 默认 c=.25。实现增补的 simplex dual，并验证 primal trust ball、simplex、
dual gap 和浮点回写后的约束。

一个公共数值尺度用于改善求解条件，不对每任务单位梯度化。零 base、零 family、零 dual vector、
solver failure、无效 simplex/certificate 都有明确状态。有限失败返回 g0；缺任务或非有限梯度
在更新前中止，避免把 NaN g0 当合法回退。没有简单地给所有退化分母加 epsilon。

所有家族与 private derivatives 在同一快照算完；检测到 closure 改参数会拒绝并恢复参数。
private 参数按自己目标更新，不随 K 机械缩小学习率；G 按用户最新要求使用自己的 geometry loss。
提供 raw/weighted block norm、cosine、g·d、实际 delta、g·delta 和同批 pre/post loss。
测试用二维角度搜索核对数值目标，并验证 Adam 实际增量与 -eta*d 不同。

这里没有声称每 API 都下降、不会遗忘或满足未经证明的全局收敛。

## J. CONTINUAL API CONTRACT

ReplayManifest/OldAPIArtifact 明确旧域训练来源、输出/家族/route/adapter 版本；拒绝 test replay。
旧版模型可冻结，只蒸馏旧 API 输出，提供连续尺度化 Huber 与明确温度的 categorical KL。
新 API 先 head-only，有限输出且优于常量参照才允许进入 joint stage。
没有全 H 蒸馏，也没有虚构 force/contact 数据或建设庞大回放服务。

## K. TESTS RUN

**最终完整 CPU suite：59 passed，0 failed，0 skipped。** 最终 pytest 报告耗时约4秒，
这是合成契约测试耗时，不是模型部署延迟/训练吞吐。详情在附录 K1 及 test-results.xml。

十二项优先验收全部覆盖：未来输入因果性、任意分块、detached-R、clean/masked 对抗泄漏、
teacher/HE 独立、family vote invariance、same snapshot、trust region、退化 fallback、
private/core 分离、valid-count normalization、resume mismatch rejection。

另检查 burn-in detach、每流隔离、适配器载入、有限缓存、六目标因果支持、EMA/统计更新边界、
缺标签 reserve、物理能量参照、几何退化告警、冻结 canonical16 数值一致和所有配置加载。
端到端 composition 测试只做一个合成参数更新，明确降低样本支持门槛作布线验收；
默认64/8/4支持规则另有独立测试，未为让生产规则过关而削弱断言。

## L. FROZEN ARTIFACT INTEGRITY

FROZEN_BASELINE.json 对原309项保护快照、5个官方checkpoint、6份设计记录和根AGENTS共321项
建立实施前快照。最终核验结果记录在 IMPLEMENTATION_EVIDENCE.json；已有文件内容均保持。
嵌套 baseline Git 工作树干净，HEAD 为 `5f6f62b1a0a08426adffe55900842e75a8adb38c`。
根不属于Git，没有创建分支或commit。核验范围不是全原始数据CRC或全磁盘审计。

## M. DEVIATIONS FROM SPEC

**架构偏离：NONE。** 工程实现选择均登记，不伪装成来源定理。
非有限 g0 的前置中止是数值定义域检查，有限求解失败仍按规格回退g0。
配置不填造真实data/statistics/reference/teacher health阈值；它们是后续实验资料，不是功能stub。

## N. BLOCKED DESIGN QUESTIONS

**NONE（没有阻断这次科学内核实现的设计问题）。**
后续真实训练仍需获准数据manifest、独立噪声依据、train统计/reference artifacts、老师预算与
健康阈值预登记。缺少它们时不允许把synthetic fixtures升级为科学训练依据。

## O. FUTURE BRANCH IDEAS DISCOVERED

NONE。本轮没有另开架构选择或实现未经批准的替代方法。

## P. REMAINING SCIENTIFIC RISKS

1. 独立EMG老师可能丢失新API信息，或保留可预测的设备/会话特征。
2. R128可能不能保留六组老师目标中的有用内容；必须比较raw/F/E/teacher/S/R/H。
3. detached-R仍允许共享F引起数值漂移；代理梯度收益不等于实际API不退化。
4. 异构API与设备/人群混杂不能由家族优化器解决；需要桥接数据和公平分组评估。
5. 真实teacher健康、跨用户迁移、force/contact/fatigue、个体化收益都尚未测量。
6. 只验证CPU float32路径；GPU/AMP、真实数据规模性能与部署延迟尚未验证。
7. replay只有契约，没有可支持旧域保留结论的真实数据或实验。

## Q. EXACT NEXT SAFE COMMAND

以下命令仅作**下一次本地合成 forward 与冻结哈希核验**；交付后未自动执行。它不训练，不读取真实数据。

```powershell
Set-Location -LiteralPath 'C:\Users\董伯言\Desktop\fingers\emg2pose_handstate'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\Anaconda3\envs\emgforce\python.exe' -m humanengine.verify --config 'humanengine/configs/he_core_v01.json' --synthetic --check-frozen
```

下一阶段建议新会话 `HUMANENGINE-VALIDATION-20260919-SCIENTIFIC-KERNEL`，先读交接记录。
没有启动老师预训练、HE_FULL、GPU/远端任务或参数扫描，也不宣称 HumanEngine 已取得科学性能。

<!-- GENERATED_APPENDICES -->

## 附录 B1. 新文件清单

- [__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/__init__.py)
- [AGENTS.md](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/AGENTS.md)
- [api.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/api.py)
- [config.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/config.py)
- [configs/he_cagrad_test.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/configs/he_cagrad_test.json)
- [configs/he_core_v01.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/configs/he_core_v01.json)
- [configs/he_family_budget_test.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/configs/he_family_budget_test.json)
- [configs/he_full_v01.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/configs/he_full_v01.json)
- [configs/he_routing_test.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/configs/he_routing_test.json)
- [configs/teacher_emg_ssl_v01.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/configs/teacher_emg_ssl_v01.json)
- [continual/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/continual/__init__.py)
- [continual/contracts.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/continual/contracts.py)
- [contracts.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/contracts.py)
- [core/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/core/__init__.py)
- [core/frontend.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/core/frontend.py)
- [core/personalization.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/core/personalization.py)
- [core/state.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/core/state.py)
- [core/summaries.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/core/summaries.py)
- [data/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/data/__init__.py)
- [data/references.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/data/references.py)
- [data/sampling.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/data/sampling.py)
- [data/sequence.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/data/sequence.py)
- [data/views.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/data/views.py)
- [diagnostics/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/diagnostics/__init__.py)
- [diagnostics/gradients.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/diagnostics/gradients.py)
- [diagnostics/representation.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/diagnostics/representation.py)
- [FROZEN_BASELINE.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/FROZEN_BASELINE.json)
- [heads/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/heads/__init__.py)
- [heads/api_heads.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/heads/api_heads.py)
- [IMPLEMENTATION_EVIDENCE.json](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/IMPLEMENTATION_EVIDENCE.json)
- [IMPLEMENTATION_MAP.md](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/IMPLEMENTATION_MAP.md)
- [IMPLEMENTATION_REPORT.md](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/IMPLEMENTATION_REPORT.md)
- [manifest.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/manifest.py)
- [objectives/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/objectives/__init__.py)
- [objectives/full.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/objectives/full.py)
- [objectives/reductions.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/objectives/reductions.py)
- [objectives/regularizers.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/objectives/regularizers.py)
- [objectives/residual.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/objectives/residual.py)
- [optimization/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/optimization/__init__.py)
- [optimization/cagrad.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/optimization/cagrad.py)
- [optimization/factory.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/optimization/factory.py)
- [optimization/macrostep.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/optimization/macrostep.py)
- [README.md](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/README.md)
- [registry.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/registry.py)
- [requirements-kernel.txt](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/requirements-kernel.txt)
- [runbooks/LOCAL_VERIFICATION.md](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/runbooks/LOCAL_VERIFICATION.md)
- [SESSION_HANDOVER.md](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/SESSION_HANDOVER.md)
- [SPEC_IMPLEMENTATION_TASK.md](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/SPEC_IMPLEMENTATION_TASK.md)
- [teacher/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/teacher/__init__.py)
- [teacher/model.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/teacher/model.py)
- [teacher/preparation.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/teacher/preparation.py)
- [test-results.xml](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/test-results.xml)
- [tests/__init__.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/__init__.py)
- [tests/conftest.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/conftest.py)
- [tests/test_core.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/test_core.py)
- [tests/test_data_provenance.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/test_data_provenance.py)
- [tests/test_integration.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/test_integration.py)
- [tests/test_objectives.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/test_objectives.py)
- [tests/test_optimization.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/test_optimization.py)
- [tests/test_teacher.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/tests/test_teacher.py)
- [verify.py](C:/Users/董伯言/Desktop/fingers/emg2pose_handstate/humanengine/verify.py)

## 附录 K1. 逐项测试记录

| Test name | Result | 验证性质 |
|---|---|---|
| `test_arbitrary_chunk_equivalence_and_tick_alignment` | PASS | 任意 chunk 分割等价，并对齐真实输出节拍 |
| `test_future_input_causality_for_H_and_API` | PASS | 改变未来输入不能改变过去 H 或 API 输出 |
| `test_reset_gap_user_adapter_and_invalid_signal` | PASS | reset gap user adapter and invalid signal |
| `test_partial_fragment_bounded_caches_and_explicit_detach` | PASS | partial fragment bounded caches and explicit detach |
| `test_raw_amplitude_precedes_personalization` | PASS | raw amplitude precedes personalization |
| `test_shadow_gradient_is_observational` | PASS | shadow gradient is observational |
| `test_missing_labels_are_not_runtime_input_and_bad_schema_fails` | PASS | missing labels are not runtime input and bad schema fails |
| `test_explicit_reset_erases_both_memories_and_caches` | PASS | explicit reset erases both memories and caches |
| `test_compatible_adapter_loading_and_reset` | PASS | compatible adapter loading and reset |
| `test_independent_stream_reset_does_not_change_other_stream` | PASS | independent stream reset does not change other stream |
| `test_ascii_HDF5_path_preserved_without_resolve` | PASS | ascii HDF5 path preserved without resolve |
| `test_canonical_frozen_mapping_and_validity_parity` | PASS | canonical frozen mapping and validity parity |
| `test_future_target_masks_boundaries_gaps_and_tail` | PASS | future target masks boundaries gaps and tail |
| `test_synchronized_source_cannot_cross_splits` | PASS | synchronized source cannot cross splits |
| `test_resume_provenance_mismatch_fails_before_mutation[route_version]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_provenance_mismatch_fails_before_mutation[teacher_hash]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_provenance_mismatch_fails_before_mutation[target_version]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_provenance_mismatch_fails_before_mutation[registry_hash]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_provenance_mismatch_fails_before_mutation[reference_scales_hash]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_provenance_mismatch_fails_before_mutation[summary_version]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_provenance_mismatch_fails_before_mutation[data_manifest_hash]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_provenance_mismatch_fails_before_mutation[statistics_hash]` | PASS | 科学版本/资料不兼容时先拒绝，模型不被部分修改 |
| `test_resume_restores_rng_model_and_sampler` | PASS | resume restores rng model and sampler |
| `test_teacher_checkpoint_resume_provenance` | PASS | teacher checkpoint resume provenance |
| `test_replay_onboarding_and_output_only_distillation` | PASS | replay onboarding and output only distillation |
| `test_representation_health_and_export_does_not_claim_physiology` | PASS | representation health and export does not claim physiology |
| `test_reference_scales_are_train_only_and_constant_predictor_based` | PASS | reference scales are train only and constant predictor based |
| `test_full_kernel_composition_and_single_synthetic_update` | PASS | 完整目标组合与一个合成更新；无标签 EMG 仍可计算 reserve |
| `test_burnin_is_detached_but_stream_values_identical` | PASS | burn-in 只截断梯度，不改变前向状态值 |
| `test_all_six_explicit_configs_load` | PASS | all six explicit configs load |
| `test_teacher_rejects_zero_mask_disguised_as_masked` | PASS | teacher rejects zero mask disguised as masked |
| `test_explicit_API_detached_R_reachability[current]` | PASS | API 数值读取 R，但监督不直接更新 R 参数 |
| `test_explicit_API_detached_R_reachability[future]` | PASS | API 数值读取 R，但监督不直接更新 R 参数 |
| `test_residual_individual_gradient_routes[masked]` | PASS | residual individual gradient routes |
| `test_residual_individual_gradient_routes[observed]` | PASS | residual individual gradient routes |
| `test_residual_individual_gradient_routes[physical]` | PASS | residual individual gradient routes |
| `test_residual_individual_gradient_routes[variance]` | PASS | residual individual gradient routes |
| `test_geometry_and_vicreg_gradient_routes[geometry]` | PASS | geometry and vicreg gradient routes |
| `test_geometry_and_vicreg_gradient_routes[vicreg]` | PASS | geometry and vicreg gradient routes |
| `test_valid_count_nan_missing_and_equal_scale_reduction` | PASS | valid-count、NA 与尺度/层等权缩减正确 |
| `test_variance_support_is_NA_and_no_neighbor_padding` | PASS | variance support is NA and no neighbor padding |
| `test_physical_summary_sinusoid_energy_and_amplitude` | PASS | physical summary sinusoid energy and amplitude |
| `test_geometry_degeneracy_is_reported_and_stops` | PASS | geometry degeneracy is reported and stops |
| `test_family_vote_invariance_E0` | PASS | 同源五份 alias 不增加家族曝光、预算和梯度贡献 |
| `test_family_fixed_denominator_missing_is_NA` | PASS | family fixed denominator missing is NA |
| `test_cagrad_trust_region_primal_dual_and_angular_reference` | PASS | trust region、dual gap 与独立二维数值参照一致 |
| `test_cagrad_aligned_and_K1_base` | PASS | cagrad aligned and K1 base |
| `test_cagrad_degenerate_fallback_missing_nonfinite_and_solver_failure` | PASS | 退化/求解失败显式处理，不掩盖非有限或缺任务 |
| `test_same_snapshot_and_private_vs_core_update_separation` | PASS | 各族同一快照，private 梯度不被家族预算缩放 |
| `test_macrostep_rejects_intra_family_parameter_update` | PASS | macrostep rejects intra family parameter update |
| `test_macrostep_reports_real_Adam_delta` | PASS | macrostep reports real Adam delta |
| `test_macrostep_NA_has_no_partial_update` | PASS | macrostep NA has no partial update |
| `test_valid_anchor_quota_plans_cap_exposure_and_report_missing` | PASS | 有效目标配额精确封顶，不能补齐时为 NA |
| `test_independent_teacher_preparation_routes_and_EMA` | PASS | 老师与 HE 隔离，EMA 只跟随自己的 student |
| `test_teacher_causal_six_targets_and_recording_support` | PASS | teacher causal six targets and recording support |
| `test_mask_protocol_and_adversarial_clean_target_leakage` | PASS | 相同 masked 输入、不同 clean target 时学生输出相同 |
| `test_teacher_health_gate_and_frozen_statistics` | PASS | teacher health gate and frozen statistics |
| `test_no_unjustified_noise_scale` | PASS | no unjustified noise scale |
| `test_frozen_statistics_never_use_per_window_or_heldout` | PASS | frozen statistics never use per window or heldout |
