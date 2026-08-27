# GPU (WebGPU)

`-tags wgpu` (linux, macOS, Windows) でビルドすると、CPU 版 `MatMul` と同じ形状・ブロードキャスト意味論の実験的 GPU バックエンドが有効になります。cgo は使いません: バインディングは [wgpu-native](https://github.com/gfx-rs/wgpu-native) 共有ライブラリを `ebitengine/purego` 経由で実行時にロードします (linux/macOS は `dlopen`、Windows は `LoadLibrary`)。wgpu-native は Linux で Vulkan、Windows で Vulkan か D3D12、macOS で Metal を選ぶので、AMD、Intel、Apple、NVIDIA の GPU がすべて動きます — lavapipe のような CPU Vulkan 実装も動き、GPU のないマシンでのテストはそれで回っています。

## セットアップ

**v22.1.0.5** のリリースバイナリ (このバインディングが対象とする C API) をダウンロードし、ローダーが見つかる場所に置くか `TENSAI_WGPU_LIB` で指定します:

```bash
curl -sLO https://github.com/gfx-rs/wgpu-native/releases/download/v22.1.0.5/wgpu-linux-x86_64-release.zip
unzip wgpu-linux-x86_64-release.zip -d wgpu
TENSAI_WGPU_LIB=$PWD/wgpu/lib/libwgpu_native.so go test -tags wgpu ./...
```

Windows では同じリリースの `wgpu-windows-x86_64-msvc-release.zip` を取り、中の `wgpu_native.dll` を変数で指定します (`PATH` 上か実行ファイルの隣にある `wgpu_native.dll` は変数なしで見つかります):

```powershell
$env:TENSAI_WGPU_LIB="$PWD\wgpu\lib\wgpu_native.dll"
go run -tags wgpu ./_example/wgpu
```

ビルドタグなしでは `gpu.Open` がエラーを返すだけで、他には何も変わりません。

## 基本的な使い方

```go
dev, err := gpu.Open() // GPU やライブラリがなければきれいに失敗する
if err != nil { /* tensai.MatMul にフォールバック */ }
defer dev.Close()
fmt.Println(dev.Name()) // 例: "AMD Radeon 780M (integrated)"
out, err := dev.MatMul(a, b)
```

iGPU と dGPU の両方があるマシンでは希望を渡せます: `gpu.Open(gpu.LowPower)` は iGPU へ、`gpu.HighPerformance` は dGPU へ寄せます (ヒントです — アダプタが 1 つならそれが返ります)。

## 常駐バッファ

`dev.MatMul(a, b)` は Upload → MatMul → Download → Free の省略形です。バッファを GPU に常駐させれば、重みはバスを 1 回しか渡らず、中間結果はデバイスを離れません:

```go
gw, _ := dev.Upload(w)              // 重みは 1 回だけアップロード
defer gw.Free()                     // GPU メモリは GC されない
gx, _ := dev.Upload(x)
h, _ := gx.MatMul(gw)               // 自由にチェーン。ホストには触れない
out, _ := h.MatMul(gw2)
result, _ := out.Download()         // 最後に 1 回だけ読み戻す
```

常駐が最も効くのは転送のたびに PCIe を渡るディスクリート GPU です。共有メモリの iGPU では効果は小さく、主に中間結果の読み戻しを省ける分になります。

## デバイス上の attention

MatMul の他に、常駐テンソルは `MatMulT`、インプレースの `Scale`、最終軸の行並列 `Softmax` をサポートします — シングルヘッド attention を GPU 上で完結させるのに十分です:

```go
out, _ := gq.Attention(gk, gv)                 // softmax(q@k^T/sqrt(d))@v、ホスト往復なし
out, _ = gq.MultiHeadAttention(gk, gv, heads)  // パックされた (batch, seq, heads*dh) レイアウト
```

マルチヘッド attention はストライド付きカーネルでパックレイアウトから各ヘッドを切り出すので、permute は一切実体化されません。causal 版 (`CausalAttention`, `CausalMultiHeadAttention`) はカーネル内で未来位置をマスクし、k と v は q より多くの位置を持てます — 自己回帰モデルのプロンプトプレフィルとチャンクデコードのパターンです — のでマスクテンソルも作られません。`CausalMultiHeadAttention` は 1 回の融合 flash-attention 風ディスパッチ (kv タイル上のオンライン softmax、ヘッド次元 128 まで) で走ります: スコア行列は存在しないので、メモリはシーケンス長によらず q+k+v+出力 のままです。

## デバイス上の量子化重み

`UploadQ8` は `QMatrix` を u32 あたり int8 重み 4 つでパックし、`gpu.QMatrix.MatMul` がレジスタ内で逆量子化するので、デコード matvec の転送は f32 の 1/4 バイトになります。`UploadQ4` は int4 の双子に同じことをするので、`-q4 -gpu` は int8 では収まらないモデルを動かせます。

トランスフォーマーのデコード 1 ステップの残りも揃っています — `RMSNorm`、インプレース `RoPE`、`Add`、`SiluMul`、`GroupedCausalAttention` (クエリよりヘッド数の少ない KV キャッシュ)、常駐キャッシュに新しい k/v 行を追記する `CopyRowsInto` — なので `_example/qwen -q8 -gpu` は全ブロックをデバイス上で実行し、トークンごとに戻ってくるのは隠れ状態だけです。`BeginBatch`/`Flush` は 1 トークン分のディスパッチを 1 回のサブミッションに記録し、解放された中間バッファはバッファプールで再利用されます。

## クロスオーバーの測定

`_example/wgpu -sweep` はサイズの階段を上りながら、GPU が CPU カーネルを追い越す地点に印を付けます。CPU 側はパッケージの他の部分と同じ `dotRows` カーネルなので、サンプルを 2 回ビルドすればポータブル Go、AVX2、GPU の 2 つの使用パターンを比較できます:

```bash
GOEXPERIMENT=nosimd go build -tags wgpu -o wgpu-nosimd ./_example/wgpu
GOEXPERIMENT=simd   go build -tags wgpu -o wgpu-simd   ./_example/wgpu
./wgpu-nosimd -sweep && ./wgpu-simd -sweep
```

Ryzen の iGPU (AMD Radeon 780M、ネイティブ Windows、AVX2 CPU カーネル) では、レジスタタイリングカーネルが階段の全段を GPU 側に載せます:

```
             shape                   MFLOP   gpu+xfer   resident        cpu   res/cpu
mnist dense  1x100x784@784x128        20.1     1.51ms      597µs      652µs     1.09x
mnist conv2  1x19600x72@72x16         45.2    1.388ms      763µs    2.354ms     3.09x
tiny         1x128x128@128x128         4.2      432µs      302µs      410µs     1.36x
small        1x512x512@512x512       268.4    1.331ms    1.216ms    6.865ms     5.65x
medium       8x512x512@512x512      2147.5    8.053ms    6.297ms    71.56ms    11.36x
large        32x512x512@512x512     8589.9    86.71ms   28.856ms  266.277ms     9.23x
huge         64x512x512@512x512    17179.9  116.374ms   62.128ms  566.726ms     9.12x
```

クロスオーバーは CPU カーネル、GPU ドライバ、転送パターンで動きます — WSL2 内の dozen のような変換レイヤー越しでは比率はほぼ同等〜3 倍に縮み、CPU Vulkan 実装では GPU パスが素直に負けます。出荷するドライバの上で測ってください。

## `-tags wgpu24`: 新しい wgpu-native API と、WSL2 の中の本物の GPU

`-tags wgpu24` は同じ `gpu.Open` API を刷新された wgpu-native C API に対してビルドします — **v29 系**のリリースバイナリと組み合わせてください。新 API の収穫は `WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter` で、非準拠の Vulkan ドライバを見えるようにします。具体的には: Mesa の dozen (Vulkan-on-D3D12) は WSL2 内で本物のホスト GPU を公開しますが、v22 API はそれを非準拠として隠して lavapipe にフォールバックします。wgpu24 ビルドなら届きます:

```bash
VK_DRIVER_FILES=/path/to/dzn_icd.json \
TENSAI_WGPU_LIB=$PWD/wgpu29/lib/libwgpu_native.so \
    go run -tags wgpu24 ./_example/wgpu   # adapter: Microsoft Direct3D12 (AMD Radeon(TM) Graphics)
```

両方のタグを付けると `wgpu24` が勝ちます。新しい方が速いわけではないことに注意: Radeon 780M の `32x512x512@512x512` で、同じシェーダが v22 ライブラリでは 85ms、v29 では 165ms です。`wgpu24` は速度のためではなく、届くアダプタのために使ってください。
