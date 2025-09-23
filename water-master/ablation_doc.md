在config.json中新添加了exclude域，该值对应了去掉第几个group训练模型。当检测到该值不为0的时候，会对模型的输入维度自动进行修改，并改变模型存储的文件夹。
对于deeponet，模型存储文件夹为`ablation/group_i`，其中i为group数，从1-16。对于lstm，模型存储文件夹为lstm_ablation/group_i

# 训练方法
## 批量生成config
create_config.py提供了批量修改config的代码，给定一个config模板，它会生成16个config文件用于跑ablation。

打开create_config.py，修改第3行为修改的模板config，修改第8行为存储config的文件夹，deeponet是`ablation/config_ablation_{i+1}.json`，lstm是`lstm_ablation/config_ablation_{i+1}.json`

## 训模型

```bash
CUDA_VISIBLE_DEVICES=1 python train.py -c ablation/config_ablation_1.json > ablation/logs/group_1.txt &

or

CUDA_VISIBLE_DEVICES=1 python train.py -c ablation/config_ablation_1.json
```
该指令去掉group1的变量训deeponet，metric在`ablation/group_1`中可以找到。

```bash
CUDA_VISIBLE_DEVICES=1 python train_lstm.py -c lstm_ablation/config_ablation_1.json > lstm_ablation/logs/group_1.txt &

or

CUDA_VISIBLE_DEVICES=1 python train_lstm.py -c lstm_ablation/config_ablation_1.json
```
该指令去掉group1的变量训lstm,metric在`lstm_ablation/group_1`中可以找到。

# 相关代码
`parse_config.py` L162 有读config自动修改的代码。

`dataset.py` L217，237，257.根据exclude的group删除读取的数据集对应的列

`lstm_dataset.py` L120.根据exclude的group删除读取的lstm数据集对应的列
