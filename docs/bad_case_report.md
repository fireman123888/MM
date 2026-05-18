# Bad Case Report

本文件用于记录失败案例和修复闭环。

## 模板

| 字段 | 内容 |
|---|---|
| Case ID |  |
| 问题 |  |
| 期望依据 |  |
| 实际回答 |  |
| 错误类型 | `retrieval_miss` / `wrong_citation` / `unsupported_claim` / `missing_facts` / `unsafe_answer` |
| 根因 |  |
| 修复动作 |  |
| 修复后结果 |  |

## 当前样例

| Case ID | 问题 | 错误类型 | 根因 | 修复动作 |
|---|---|---|---|---|
| BC-001 | “我工作8个月没签劳动合同，月薪8000，可以赔多少？” | retrieval_rank | 第八十二条曾排在第四十七条后面。 | 增加未签合同和二倍工资短语 boost。 |
| BC-002 | 多轮第二问“月薪8000，能赔多少？” | missing_context | 第二轮需要继承第一轮工作时长。 | 增加 session facts 合并和 retrieval query 扩展。 |
