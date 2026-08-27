# SIMD アクセラレーション

tensai の高速カーネルは AVX2 で、Go の実験的な `simd/archsimd` パッケージで書かれています — 純 Go のまま、cgo なし、アセンブリファイルなしです。

```bash
GOEXPERIMENT=simd go build ./...
GOEXPERIMENT=simd go test -bench=Dot .
```

要件は amd64 と Go 1.26 または 1.27 (両世代の `simd` API にビルドタグで対応)。それ以外のビルド — 他アーキテクチャ、古い Go、`GOEXPERIMENT` 未設定 — は自動的にポータブル実装へフォールバックし、結果は同一です。

## ベクトル化の範囲

AVX2 カーネルが今日適用されている場所と、まだ適用できる場所:

- [x] Matmul (`Dot`/`DotInto`) — `Dense`、`Conv2D` (im2col 積)、`KNN` 距離、自動微分 `MatMul` が使用
- [x] ReLU / LeakyReLU の順伝播と逆伝播
- [x] Sigmoid / Tanh の順伝播と逆伝播 (ベクトル化した多項式 `exp`)
- [x] GELU の順伝播と逆伝播 (ベクトル化した `erf`)
- [x] LayerNorm の順伝播と逆伝播 (行のベクトルリダクション)
- [x] Softmax / SoftmaxCrossEntropy の指数計算とスケーリング
- [x] Adam / AdamW のパラメータ更新
- [x] スライスの加算/スケールプリミティブ (バイアス加算、`Embedding` 勾配の scatter-add)
- [x] 転置不要の勾配 matmul (`DotTAInto`) — `Dense`/`Conv2D` の重み勾配は `input^T` / `im2col^T` を実体化しません
- [x] 残りの転置 (`T`/`TInto`) — キャッシュブロッキングされた 32x32 タイル
- [x] Softmax 逆伝播の行内積 (自動微分) — 融合 AVX2 内積とヤコビアン・ベクトル累積
- [ ] MSE / BinaryCrossEntropy 損失 (BCE はベクトル化 `log` が必要)
- [ ] 自動微分の要素ごと逆伝播 (勾配が `+=` で累積するため専用の融合カーネルが必要)
- [ ] BatchNorm の統計量 (列ストライドアクセスの再構成が必要)
- [ ] MaxPool2D のウィンドウ走査
- [ ] im2col / col2im の gather-scatter (連続区間はバルクコピーにできる)
- [ ] SGD 更新

未チェックの項目はおおよそ期待効果順ですが、どれも今の学習プロファイルでは目立ちません。

int8/int4 量子化 matmul は 256 ビットの u8 x s8 ペア積和に基づく独自の AVX2 パスを持ちます — [量子化](quantization.md)参照。

## 生きたベンチマーク

`_example/plasma` はターミナルにデモシーン風のプラズマをアニメーションします。プラズマ関数はランダムに重み付けされたネットワーク (CPPN) で、毎フレーム全ピクセルを 1 バッチとして評価します。ステータス行にフレームあたりのネットワーク時間が出ます: 120x90 ピクセルがポータブルビルドで約 32 fps、同じマシンの `GOEXPERIMENT=simd` で約 100 fps。

```bash
go run ./_example/plasma
GOEXPERIMENT=simd go run ./_example/plasma
```
