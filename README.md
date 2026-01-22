## 使用说明

### 数据格式
--dataset_test \
\t --test1 \
\t \t--input.png \
\t \t--gt.png \
\t \t--\{model\}-output.png \
prompt.csv fields:
editing_prompt,reference_phenomenon

### 测评方式
首先使用qw_checklist_test.py生成checklist
其次使用qw_vlm_test.py生成score

### 结果
checklist结果保存在checklist.json
score结果保存在score.json
