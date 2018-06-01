# README.txt

# seq2seq_keras_based.py
- 个人实现基于Keras的seq2seq，运行命令：
$> python seq2seq_keras_based.py

# nmt
- 开源TensorFlow NMT实现，运行命令：
- seq2seq：
$> python -m nmt.nmt --num_gpus=4 --src=en --tgt=ru --vocab_prefix=rus-eng/vocab --train_prefix=rus-eng/train --dev_prefix=rus-eng/dev --test_prefix=rus-eng/test --out_dir=nmt_model --num_train_steps=12000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=bleu
$> python -m nmt.nmt --out_dir=nmt_model --inference_input_file=rus-eng/infer_file.en --inference_output_file=rus-eng/nmt_model_output_infer.ru

- seq2seq(attention)：
$> python -m nmt.nmt --attention=scaled_luong --num_gpus=4 --src=en --tgt=ru --vocab_prefix=rus-eng/vocab --train_prefix=rus-eng/train --dev_prefix=rus-eng/dev --test_prefix=rus-eng/test --out_dir=nmt_attention_model --num_train_steps=12000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=bleu
$> python -m nmt.nmt --out_dir=nmt_attention_model --inference_input_file=rus-eng/infer_file.en --inference_list=184 --inference_output_file=rus-eng/nmt_attention_model_output_infer.ru

#GIZA++
- 开源工具，运行命令：
$> ./snt2cooc.out vocab.en vocab.ru train.en2ru > en2ru.cooc
$> ./GIZA++ -S vocab.en -T vocab.ru -C train.en2ru -CoocurrenceFile en2ru.cooc
