# Shortcut DFA 大函数表的分层分解：内存—时间边界的实证刻画

基于 Shortcut2Secrets（TCHES 2025）的开源实现 https://github.com/SJTUwwz/DFA_Elisabeth_family，
在 Elisabeth-4 / Elisabeth-b4 上对比过滤表 T_{δ,j} 的几种组织方式（均匀 15×4、倒排索引、NLUT 对称压缩）。
一个为考研复试准备的小项目，由于时间紧，不少部分用了claude编程QAQ。

- 结果与结论：[experiments/RESULTS.md](experiments/RESULTS.md)
- 运行前把原仓库克隆到本目录：`git clone https://github.com/SJTUwwz/DFA_Elisabeth_family`
- 依赖：Python 3、numpy、psutil
