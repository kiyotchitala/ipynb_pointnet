# pointnet実装の結果

データセットは<br>
・人間<br>
・木

高さは95cmと40cmとしている<br>
<img src='./pointnet_tree_human_45_90.png' width=400><br>


高さは95cmのみとしている<br>
きちんと人間と木の判別ができていることが分かる
<br>
Accuracyとは，予想がどれだけ正しかったかを見る指標である<br>
<img src='./pointnet_human_tree_95.png'><br>
損失関数も下がりつつあり，モデルはきちんと構成されていると言える<br>
精度も100%に張り付いている<br>
っしゃ！！