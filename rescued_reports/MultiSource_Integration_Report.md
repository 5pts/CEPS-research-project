# 多源数据整合与处理报告

## 1. 数据整合概况
本报告记录了将家长、教师、学校数据整合至学生主数据集（Rescued Student Data）的过程与结果。

### 1.1 数据源清单
*   **学生数据**：基准数据集，N=9763 (清洗后)
*   **家长数据**：补充家庭背景信息 (Income, Expectation)
*   **教师数据**：补充班级环境信息 (Teacher Demographics)
*   **学校数据**：补充学校层级信息 (School Type, Location)

## 2. 处理逻辑

### 2.1 家长数据 (Parent Data)
*   **匹配键**：`ids` (学生唯一ID)
*   **提取变量**：
    *   `be01` -> `parent_income`: 家庭收入 (自报)
    *   `be13` -> `parent_expect`: 父母对子女的教育期望
*   **处理**：直接左连接 (Left Join) 到学生数据。

### 2.2 教师数据 (Teacher Data)
*   **匹配键**：`clsids` (班级ID)
*   **提取变量**：
    *   `hr01`: 教师性别
    *   `hr02`: 教师年龄
*   **处理**：
    *   由于一个班级可能有多个科任老师，目前策略为**保留该班级的第一条教师记录**（通常为班主任或主科老师），作为班级师资特征的代理变量。
    *   通过 `clsids` 将教师特征聚合到学生层级。

### 2.3 学校数据 (School Data)
*   **匹配键**：`schids` (学校ID)
*   **提取变量**：
    *   `pla01`: 学校位置 (城市/农村)
    *   `pla04`: 学校性质 (公立/私立)
*   **处理**：直接左连接到学生数据，用于构建 HLM 模型的 Level-2 变量。

## 3. 整合结果

*   **最终数据集**：`merged_rescued_all.csv`
*   **样本量**：9763 (保持学生基数不变)
*   **新增变量**：
    *   `parent_income`, `parent_expect`
    *   `hr01` (Teacher Gender), `hr02` (Teacher Age)
    *   `pla01` (School Loc), `pla04` (School Type)

## 4. 质量提示
*   **教师匹配率**：部分班级可能缺失对应的教师问卷数据，导致 `hr01/02` 存在缺失值。建议在 HLM 建模时检查 Level-2 缺失情况。
*   **家长一致性**：可对比 `parent_expect` 与学生自报的 `expect_college`，检验家庭内部期望的一致性。
