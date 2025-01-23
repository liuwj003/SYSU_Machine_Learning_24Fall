```
Code文件夹里最后删去了原始的csv文件，要运行代码则要把csv文件放进来
参考command如下：
python main.py --learning_rate 1e-2 --model "linear" --loss_func "cross-entropy" --regular_strength 1e-3 --epochs 100
python main.py --learning_rate 1e-3 --model "linear" --loss_func "hinge" --regular_strength 1e-3 --epochs 100
python main.py --model "SVM" --kernel_func "Gaussian" 
python main.py --model "SVM" --kernel_func "Linear" 
```
