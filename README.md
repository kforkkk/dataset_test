## 使用说明

### 数据格式
```
dataset_test/
  ├─ test1/
  │   ├─ input.png                  # 输入图像（原图或需要编辑的图）
  │   ├─ gt.png                     # 目标参考图（ground truth，可选）
  │   ├─ modelA-output.png          # 模型输出：命名格式为 {model_name}-output.png
  │   └─ modelB-output.png          #（可以有多个不同模型的输出）
  ├─ test2/
  │   ├─ input.png
  │   ├─ gt.png
  │   └─ mymodel-output.png
  └─ prompt.csv                      # 每行对应一个测试样例的提示信息（见下）
```
prompt.csv fields: \
editing_prompt,reference_phenomenon

### 测评方式
首先使用qw_checklist_test.py生成checklist \
其次使用qw_vlm_test.py生成score \

### 结果
checklist结果保存在checklist.json \
score结果保存在score-\{model\}.json \
