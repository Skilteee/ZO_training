# ZO_training

牛老师好，所需要的数据都已经上传，您应该唯一需要额外下载的就是model本身的参数.

一共2个training文件，FP_training.py, Quant_training.py. 可以直接运行: CUDA_VISIBLE_DEVICES=0 python FP_training.py 或 Quant_training.py.

使用的transformers的版本为4.38.2.

与FP training相关的script就只有FP_training.py和FP_zo_trainer.py， 其他的script都是和Quant_training相关的。

根据我自己的测试, 不修改其中参数的情况下运行FP_training.py进行training的memory占用是5942MB, 基本相当于只存储一个FP16的OPT2.7B模型.
运行Quant_training进行training的memory占用是6102MB.


update - 0919:

quantize/quantizer_pack.py中有在fine-tuning中把weight进行pack和quantize从而减少memory的code. 在fine-tuning时，除了model的W_q和W_v是原始的weight，其他的weight都是int形式保存，然后forward时候unpack和dequant一下. 

example model after packing (OPT-2.7B): https://drive.google.com/file/d/11n2rVIJfCaOP6u9FMpmL0s3Zbeotlidv/view?usp=sharing

