https://github.com/tensorflow/models をフォーク

reserach/deeplab/を使用

学習方法
1. cvat で `actions > Export as a dataset > Segmentation mask 1.1`を選択してデータをダウンロードして解凍
1. 解凍したデータの中身を`reserach/deeplab/datasets`に配置
1. `datasets/road_heating/make_trainval.py` を実行
    - `road_heating/Imagesets/Segmenntation`にtrain.txt, val.txt, trainval.txtがあることを確認
1. `deeplab/utils/get_dataset_colormap.py`を修正 (340行目くらい)
    - カラーマップをcvatで設定したものにする(1/7時点ではcvatのカラーマップに準拠)
```python:get_dataset_colormap.py
colormap = np.array([
    [0,0,0], # Background
    [50,183,250], # Snow,
    [255, 96,55], # Road
    [131,224,112], # Obstacle
])

return colormap
```

1. `remove_gt_colormap.py `を実行
    - `road_heating/SegmentationClassRaw`が出来ていることを確認
1. 以下を実行
```bash
mkdir road_heating/tfrecord
python build_voc2012_data.py --image_format="jpg"
```
6. `data_generator.py`を修正 (102行目くらい)
    - train, trainval, valの数字をそれぞれのデータ数に変更 (make_trainval.pyの実行時にそれぞれの数が出力される)
```python:data_generator.py
_ROAD_HEATING_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 190,
        'trainval': 152,
        'val': 38,
    },
    num_classes=4,
    ignore_label=255,
)
```
7. パスを通す．reserach/で以下を実行.起動のたびに必要なので，面倒だったら~/.bashrcに記載
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

7. 以下を実行して学習 
    - --train_log_dirにチェックポイントがあるとそれを読み込んでしまうので，何回も学習する場合は別のディレクトリを指定する必要あり
    - --training_number_of_stepsで学習回数変更(epochではないことに注意)

```
python train.py --logtostderr --training_number_of_steps=1000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=4 --dataset="road_heating" --train_logdir="./datasets/road_heating/exp/train_on_trainval_set/train" --dataset_dir="./datasets/road_heating/tfrecord" --fine_tune_batch_norm=false --initialize_last_layer=true --last_layers_contain_logits_only=false
```
8.  評価
```
python eval.py   --logtostderr   --vis_split="val"   --model_variant="xception_65"   --atrous_rates=6   --atrous_rates=12   --atrous_rates=18   --output_stride=16   --decoder_output_stride=4   --vis_crop_size="513,513"   --checkpoint_dir="./datasets/road_heating/exp/train_on_trainval_set/train"   --eval_logdir="./datasets/road_heating/exp/train_on_trainval_set/eval"  --dataset_dir="./datasets/road_heating/tfrecord"   --max_number_of_iterations=1 --dataset=road_heating
```
9. 可視化　(`road_heating/exp/vis/segmentation_results`に出力)
```
python vis.py   --logtostderr   --vis_split="val"   --model_variant="xception_65"   --atrous_rates=6   --atrous_rates=12   --atrous_rates=18   --output_stride=16   --decoder_output_stride=4   --vis_crop_size="513,513"   --checkpoint_dir="./datasets/road_heating/exp/train_on_trainval_set/train"   --vis_logdir="./datasets/road_heating/exp/train_on_trainval_set/vis"  --dataset_dir="./datasets/road_heating/tfrecord"   --max_number_of_iterations=1 --dataset=road_heating
```

引数の説明とかも参考にのってる
## 参考
- https://qiita.com/mine820/items/7a08fad3847f2981cb01
- https://qiita.com/mucchyo/items/d21993abee5e6e44efad
- https://qiita.com/harmegiddo/items/b11decca0769fc108ec9
