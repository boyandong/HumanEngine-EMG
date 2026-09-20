# HUMANENGINE KERNEL REMEDIATION REPORT

2026-09-19。范围仅 humanengine/；证据均为本地合成 CPU 正确性检查。

## 1. STATUS

**READY FOR INDEPENDENT RE-AUDIT**。
这不是 READY FOR TEACHER VALIDATION。未训练真实 teacher、HE_FULL，未启动 GPU、远端或 sweeps。
F06 已完成只读核对，但具体授权／归因仍 UNRESOLVED；该状态不被测试通过覆盖。

## 2. FINDINGS MATRIX

| ID | Severity | Root cause | Fix | Regression | Status |
|---|---|---|---|---|---|
| F01 | BLOCKER | 仅依赖 _version，回滚只覆盖参数 | 比较真实内容；保护参数、梯度、优化器、controller、注册模块及 sampler；拒绝时回滚 RNG | data/private mutation、非空 Adam、optimizer-only、sampler、buffer/metadata、pure closures | 本地回归通过，待独立复审 |
| F02 | MAJOR | floor/provenance/adapter 不在 checkpoint 语义中，统计 hash 仅是声明 | v2 内容校验、真实统计 hash、runtime 定义核对、metadata 回滚 | mean/std tamper、重算外层 seal 仍拒绝、floor/provenance/adapter/mode、teacher、晚期失败 | 本地回归通过，待独立复审 |
| F03 | MAJOR | health 未绑定实际 EMA 和统计 | v2 artifact certificate；导出及 FrozenTeacher 构造核验 | unchanged、EMA/target/physical/config/manifest/step mutation、直接构造 | 本地回归通过，待独立复审 |
| F04 | MAJOR | runtime warmup 被当作 training eligibility | 显式 post-burn-in eligibility，anchor<B 拒绝 | B−1 拒绝，tick 对齐的 B 接受且 core 梯度非零，warmup≠burn-in | 本地回归通过，待独立复审 |
| F05 | MAJOR | 重访窗口按 count 重付开头锚点 | explicit AnchorIdentity、consumed 集合、精确 index grants | 5<7 返回 NA，多窗恰好 7，aliases/retries 无重复 | 本地回归通过，待独立复审 |
| F06 | MAJOR / provenance UNKNOWN | 冻结历史与当前 S6 状态不同，缺授权链 | 只读报告及前后快照，原基线不变 | 321 项 entry/final hash 对比 | 调查完成；9 项来源 UNRESOLVED |
| F07 | MINOR | 用 σ² 概率计算 entropy | σ 概率；定义 v2；零谱=0 | 3:1 解析例 | 本地回归通过，待独立复审 |
| F08 | MINOR | FP32 norm 在转 double 前溢出 | critical norms、geometry、gains/certificate 使用 float64 | 有限 1e20 FP32 输入 | 本地回归通过，待独立复审 |
| F09 | Clarification | 私有 M_A 系数没有明确契约 | explicit physical_weight×L_phys，默认 .05，不乘 alpha | alpha=.2/.4 的实际私有梯度 | 明确并通过回归 |

## 3. FILES MODIFIED

源码：api.py、config.py、manifest.py；optimization/{macrostep,factory,cagrad}.py；
teacher/{preparation,model}.py；data/sampling.py；objectives/full.py；
diagnostics/representation.py。新增 scientific_state.py。

测试：tests/{conftest,test_data_provenance,test_integration,test_optimization,test_teacher}.py；
新增 tests/test_remediation.py。没有删除旧测试。

六份 configs/*.json 明确 M_A private rule。文档更新 README、IMPLEMENTATION_REPORT、
IMPLEMENTATION_MAP、SESSION_HANDOVER。新增 REMEDIATION_NOTES、此报告、F06_PROVENANCE_RECONCILIATION、
REMEDIATION_PROTECTED_BEFORE/AFTER.json、REMEDIATION_REPRODUCTIONS_BEFORE.json、
REMEDIATION_EVIDENCE.json、remediation-test-results.xml、runbooks/remediation_evidence.py。
完整文件内容 hash 在 machine evidence 中。FROZEN_BASELINE.json 与历史 test-results.xml 未重写。

## 4. F01 SAME-SNAPSHOT FIX

修复前重新运行独立反例：`.data.add_(1)` 未拒绝且 steps=1；Adam.step 被拒绝但留下 moment/state。
旧测试仅覆盖会改变 _version 的 no_grad.add_，不能发现这个漏洞。

现在入口克隆真实 core/private 张量与已有梯度、所有所属 optimizer state、controller、RNG；
对每个 closure 返回后及 autograd 后都核对内容和状态。factory 注册 model/heads，保护
其冻结参数、buffers 和语义 metadata；有 sampler 的调用必须显式传 sampler=。
拒绝后恢复全部这些已登记状态，不执行科学更新。纯 closure 的所有梯度与同一参数内容的
独立 autograd 结果一致。新增回归包括已有非空 Adam state 与仅改 moment 的情况。

限制：这是注册状态的正确性边界，不是任意 Python 副作用的安全沙箱；外部文件、远端进程、
未登记全局对象不在控制范围。独立调用 Macrostep 必须声明 stateful_modules 和 sampler。

## 5. F02 CHECKPOINT / STATISTICS SEMANTICS

修复前相同 signature 恢复接受 floor .001→.5、不同 provenance/adapter；E/H 数值改变。
E.mean 被篡改也获接受。teacher 物理摘要同样接受不兼容 floor。

he-kernel-checkpoint-v2 与 emg-teacher-preparation-v2 使用确定性类型/shape/原始 tensor
bytes 哈希，含全部序列化 payload；统计另有真实内容 hash。HE 绑定 mean/std/floor/provenance/
definition、summary version、input schema、所有 module config、E window、adapter identity/version、
mode/trainability、route/config；A/U 张量在 sealed model state 中。teacher 绑定物理统计及运行
target 统计定义；target moments/count 和 step 在内容校验的 state 中。

恢复先重算 actual content、核对 declared signatures，再对比 runtime 定义及真实统计内容，
之后才加载。冻结统计不同必须拒绝；不是自动迁移。v1 不静默兼容。晚期加载失败回滚
model/heads/optimizers/controller/sampler/RNG 以及 plain metadata、adapter 身份、trainability。
已用注入的晚期 RNG 失败与 metadata 污染加载失败检验回滚。构造 checkpoint 时假统计 hash 也拒绝。

限制：checksum 保证一致性，不验证数据 manifest 声明是否真实；真实来源审计仍需要后续实验记录。

## 6. F03 TEACHER ARTIFACT BINDING

修复前能复用 health/hash 导出全零 EMA。现在 teacher_artifact_identity 绑定 EMA 参数/buffers、
architecture/module config、target/physical statistics、data manifest、training step/checkpoint ID、
target/preparation version、evaluation config/version、partition/budget identity 和登记阈值。
health.artifact_identity 必须来自被评估的那个 artifact。导出重新计算并严格相等；直接
FrozenTeacher 构造也验证 certificate 与真实 encoder/statistics。

不变 artifact 通过；变更 EMA、target、physical、evaluation、manifest 或 step 均拒绝。
测试 fixture 明确是 synthetic wiring；未编造真实健康阈值，未进行真实 teacher 质量评估。
限制：证书认证“哪些内容被声明评估过”，不能证明人为提交的 metric 数字是真实测量。

## 7. F04 BURN-IN FIX

修复前 default anchor1999 有非零 pose/residual loss，但全部 core 梯度为零。
现在训练输出另带 training_eligible mask；full_objectives 显式要求 raw anchor>=B，
同时保留 runtime warmup 与 output tick 检查。0..B−1 不得计分。

默认 B=2000 时 2000 本身不是 stride40 output tick；不会凭空创建输出，首个可用 tick 为2039。
回归另设 B=3999，使 B 本身是 tick，验证它被接受且 pose 对 core 的梯度非零。
warmup=.2/1.0 秒均覆盖，运行有效性与训练资格不再混淆。

## 8. F05 QUOTA FIX

修复前单窗5个有效锚点，quota7 通过重访变成0,1,2,3,4,0,1。
修改旧测试前已在 REMEDIATION_NOTES 解释其错误期待：重复支付不能补齐有效曝光。

scientific-anchor-v2 使用 source/user/session/recording/window/index/component 身份。
coverage 必须提供实际有效 index 列表，整数 count 被拒绝；每个计划有 consumed 集合。
quota_mask 只选择获批 index，不再取前 N 个 valid。5<7 为 NA；多窗有足够支持时恰好
授予7个唯一身份；复制登记及重访不能重复支付。

限制：真实数据层仍须使用可信稳定的 acquisition/window ID；本任务不构建真实数据采样流水线。

## 9. F06 PROVENANCE RECONCILIATION

详见 F06_PROVENANCE_RECONCILIATION.md，逐路径记录证据及限制。
独立审计8项，本次入口9项：多出 logs/staged_site_manifest.json，发生在本次源码编辑前。
9项均保守标为 UNRESOLVED；S6后续演化记录不足以认证确切旧/新字节与授权。
未读隐藏答案，未恢复文件，未重写旧冻结基线。

## 10. F07 EFFECTIVE RANK

原架构 K2 明确 p=σ/Σσ；修正前用 covariance eigenvalue σ²。
现在 centered-singular-value-entropy-v2 采用登记定义；covariance_spectrum 仍单独保留。
3:1例预期1.7547653506033232，旧结果1.384145488461686，修复回归匹配前者；零谱为0。
没有改变 std threshold 或其他健康阈值。高秩不证明生理信息有效。

## 11. F08 CAGrad NUMERICS

修复前有限 FP32 1e20 输入触发 zero_dual fallback，trust residual=-inf。
现在先转 float64 计算 critical norms、common scale、geometry、direction gains 和 cast certificate。
该例正常 COORDINATED，方向、gains、trust residual、dual gap 有限，并满足相对精度下球约束。
普通数学、alpha、求解器和旧解析测试未削弱；仍允许真实退化/求解失败的明确 base fallback。
限制：结果写回原参数 dtype，真正不可表示的方向仍必须拒绝／回退；本例不是所有 dtype 的保证。

## 12. F09 M_A SEMANTICS

v0.1：M_A private loss=physical_weight×L_phys，默认.05；不乘reserve_alpha。
这等价于旧 residual.total 对 M_A 的导数，现在直接写出目标以消除歧义。
config.ma_private_rule 与六份配置记录规则；alpha=.2/.4 下 actual private gradient 都等于
.05×raw physical gradient。物理摘要仍是弱锚，没有升为主 residual target。

## 13. REGRESSION TESTS

tests/test_remediation.py 包含37项参数化回归。按 F01、F02、F03、F04、F05、F07、F08、F09
命名；F06 为只读 hash/provenance 证据。修复前独立复现另存 JSON，未改写审计目录产物。
原59项仍在；配额旧错误期待已明确纠正，checkpoint/health fixture 更新为真正内容身份。
没有降低 valid causality/routing/math assertions 或扩大其 tolerances。

开发期首轮91通过/1失败：新增 burn-in fixture 使用要求最长窗的默认 mask builder，
在进入被测路径前失败；改用审计显式 mask 构造。随后94通过；最后补2项状态污染回归，96通过。

## 14. FULL TEST SUITE RESULT

**96 passed / 0 failed / 0 errors / 0 skipped；13.113秒。** 原59项 + 新37项。
环境 D:\Anaconda3\envs\emgforce\python.exe；CPU。JUnit：remediation-test-results.xml，
SHA256 a5f002693d50fda982fa1e9c158d433bf1deb0d749a976e2c464f6c683c07f0b。
禁用 bytecode/cacheprovider，临时输出只在 humanengine。没有安装环境或运行真实数据训练。

## 15. PROTECTED ARTIFACT INTEGRITY

登记的321项 entry→final全部一致，新增差异0。对历史 frozen baseline 是312匹配、8变化、
1缺失；两种比较不可混淆。原 baseline 文件未更新。范围是登记保护集，不声称全盘／raw数据CRC。
嵌套 baseline Git clean，HEAD 5f6f62b1a0a08426adffe55900842e75a8adb38c；根目录非Git。
精确前后值、source/config hashes、test artifact hash、版本、finding状态见 REMEDIATION_EVIDENCE.json。

## 16. REMAINING SCIENTIFIC RISKS

独立复审尚未进行；真实teacher质量、跨用户/新API信息保留、个体化、GPU/AMP、部署性能未验证。
F06来源仍未知；checksum不能替代真实数据授权和健康测量来源。真实调用方必须正确登记可变
训练对象及稳定anchor IDs。无架构重选，A→F→U、detached-R、six teacher targets、residual公式、
family budgeting与CAGrad数学保持。没有把合成检查升级为科学性能结论。

## 17. EXACT NEXT SESSION

**HUMANENGINE-REAUDIT-20260919-KERNEL-REMEDIATION**，使用新的独立审计者。
先读 root/lane AGENTS、SESSION_HANDOVER、此报告、REMEDIATION_EVIDENCE、原审计及原规范。
重点独立攻击 F01注册状态边界、F02语义/事务回滚、F03陈旧证书、F04边界与F05身份去重；
保持S6盲法，未获新授权不得训练。当前修复阶段到此停止；旧任务保留作阶段记录。
