https://github.com/tensorflow/models をフォーク


環境
133.87.136.2のkambeのホームディレクトリに`sing-tensorflow.sif`というイメージファイルがあるのでこれを自分のホームディレクトリ等にコピーして使ってください(tensorflow-gpu==1.15.0の環境)

下記のようにすることで，実行できる
```
singularity exec --nv ~/sing-tensorflow.sif python **.py
```
~/.bashrcに
```
alias sing='singularity exec --nv ~/sing-tensorflow.sif'
```
のようにaliasを設定しておくと楽

参考:[singularityの使い方](https://harmony-lab.esa.io/posts/6)

以下では，singularity コンテナで作業を行う．（シンギュラリティ部分は省略）

<br>

reserach/deeplab/を使用

学習方法
1. cvat で `actions > Export as a dataset > Segmentation mask 1.1`を選択してデータをダウンロードして解凍
1. 解凍したデータの中身を`reserach/deeplab/datasets`に配置
    - ImageSets/Segmentation/default.txtは削除(ディレクトリは残す)
1. パスを通す．reserach/で以下を実行.起動のたびに必要なので，面倒だったら~/.bashrcに記載
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
4. `datasets/road_heating/make_trainval.py` を実行
    - `road_heating/Imagesets/Segmenntation`にtrain.txt, val.txt, trainval.txtがあることを確認
1. `deeplab/utils/get_dataset_colormap.py`を修正 (340行目くらい)
    - カラーマップをcvatで設定したものにする(1/7時点ではcvatのカラーマップに準拠)
```python:get_dataset_colormap.py
colormap = np.array([
    [0,0,0], # Background
    [50,183,250], # Snow,
    [255,96,55], # Road
    [131,224,112], # Obstacle
])

return colormap
```

6. `remove_gt_colormap.py `を実行
    - `road_heating/SegmentationClassRaw`が出来ていることを確認
1. 以下を実行
```bash
mkdir road_heating/tfrecord
python build_voc2012_data.py --image_format="jpg"
```

この時点で以下のようにデータが配置されているはず

```
road_heating/
│
├─ImageSets
│  └─Segmentation
│          train.txt
│          trainval.txt
│          val.txt
│
├─JPEGImages
│      001.jpg
│      002.jpg
│      003.jpg
│
├─SegmentationClass
│      001.png
│      002.png
│      003.png
│
├─SegmentationClassRaw
│      001.jpg
│      002.jpg
│      003.jpg
│
└─tfrecord
        001.tfrecord
        002.tfrecord
        003.tfrecord
```
8. `data_generator.py`を修正 (102行目くらい)
    - train, trainval, valの数字をそれぞれのデータ数に変更 (make_trainval.pyの実行時にそれぞれの数が出力される)
    - 学習済みモデルを使うので，num_classesはそのまま
```python:data_generator.py
_ROAD_HEATING_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 285,
        'trainval': 228,
        'val': 57,
    },
    num_classes=21,
    ignore_label=255,
)
```



9. 学習済みモデルを入手
```bash
mkdir road_heating/init_models/
wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
mv deeplabv3_pascal_train_aug_2018_01_04.tar.gz road_heating/init_model
tar -xvf deeplabv3_pascal_train_aug_2018_01_04.tar.gz
```

10. 以下を実行して学習 
    - --train_log_dirにチェックポイントがあるとそれを読み込んでしまうので，何回も学習する場合は別のディレクトリを指定する必要あり
    - --training_number_of_stepsで学習回数変更(epochではないことに注意)

```
sing python train.py --logtostderr --training_number_of_steps=1000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18  --decoder_output_stride=4 --train_batch_size=4 --dataset="road_heating" --train_logdir="./datasets/road_heating/exp/train_on_trainval_set/train" --dataset_dir="./datasets/road_heating/tfrecord" --fine_tune_batch_norm=false --initialize_last_layer=true --last_layers_contain_logits_only=false  --tf_initial_checkpoint="./datasets/road_heating/init_models/deeplabv3_pascal_train_aug/model.ckpt"
```
11.  評価(多分終了しないので，miouが出力された時点で終了する)
```
sing python eval.py   --logtostderr   --model_variant="xception_65"   --atrous_rates=6   --atrous_rates=12   --atrous_rates=18   --output_stride=16   --decoder_output_stride=4   --checkpoint_dir="./datasets/road_heating/exp/train_on_trainval_set/train"   --eval_logdir="./datasets/road_heating/exp/train_on_trainval_set/eval"  --dataset_dir="./datasets/road_heating/tfrecord"   --max_number_of_iterations=1 --dataset=road_heating
```
12. 可視化　(`road_heating/exp/vis/segmentation_results`に出力)
```
sing python vis.py   --logtostderr --model_variant="xception_65"   --atrous_rates=6   --atrous_rates=12   --atrous_rates=18   --output_stride=16   --decoder_output_stride=4   --checkpoint_dir="./datasets/road_heating/exp/train_on_trainval_set/train"   --vis_logdir="./datasets/road_heating/exp/train_on_trainval_set/vis"  --dataset_dir="./datasets/road_heating/tfrecord"   --max_number_of_iterations=1 --dataset=road_heating
```

引数の説明とかも参考にのってる
## 参考
- https://qiita.com/mine820/items/7a08fad3847f2981cb01
- https://qiita.com/mucchyo/items/d21993abee5e6e44efad
- https://qiita.com/harmegiddo/items/b11decca0769fc108ec9
