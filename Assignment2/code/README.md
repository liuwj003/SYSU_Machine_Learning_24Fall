````
参考command：
python main.py --learning_rate 1e-4 --model "Softmax" --epochs 200 --optimizer "SGD"
python main.py --model "MLP" --epochs 50 --optimizer "Adam"
python main.py --model "CNN5" --epochs 30 --optimizer "Adam"
可视化的loss和accuracy结果放在了plot_result文件夹中