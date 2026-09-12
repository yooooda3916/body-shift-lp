#!/usr/bin/env python3
"""
LP用画像生成スクリプト — image-create.py のLP特化版
Gemini API を使用してLP向けの画像素材を生成する

既存の image-create.py との違い:
  - LP向けプリセット: hero / portrait / section-bg / icon / ogp
  - スタイル: photo-realistic / warm / professional / minimal / bright
  - テキストは入れない前提 — HTMLで重ねるため
  - レスポンシブ対応: CSS background-image として使う想定

使用例:
    # ヒーロー画像 — 16:9 ワイド、写実的
    python lp-image-create.py --preset hero --prompt "自宅リビングで筋トレする40代男性" --output hero-bg.png

    # ポートレート — 1:1 正方形、人物写真風
    python lp-image-create.py --preset portrait --prompt "笑顔のフィットネストレーナー" --output trainer.png

    # セクション背景 — 16:9、抽象的・テクスチャ風
    python lp-image-create.py --preset section-bg --prompt "健康的で明るい雰囲気の抽象背景" --output bg-cta.png

    # OGP画像 — 1.91:1
    python lp-image-create.py --preset ogp --prompt "フィットネスの成果を感じる40代男性" --output ogp.png

    # プリセットなし — 自由指定
    python lp-image-create.py --prompt "ダンベルとヨガマット" --aspect-ratio 4:3 --style minimal --output equipment.png

環境変数 — body-shift-lp/.env に書けばプロジェクト単位で設定できる:
    GOOGLE_CLOUD_PROJECT: Vertex AI プロジェクトID — 設定時はVertex AI優先
    GOOGLE_CLOUD_LOCATION: リージョン — gemini-3-pro-image を使うには global が必須
    GEMINI_API_KEY: Google Gemini API キー — Vertex AI を使わない場合
"""

import argparse
import io
import os
import json
from pathlib import Path
from typing import Optional

from PIL import Image
from google import genai
from google.genai import types


# =============================================================================
# 環境変数読み込み
# =============================================================================

def load_env_file() -> None:
    """スクリプト近傍の .env を探して環境変数に読み込む"""
    for start_dir in [Path(__file__).parent, Path.cwd()]:
        current = start_dir.resolve()
        while current != current.parent:
            env_path = current / ".env"
            if env_path.is_file():
                print(f".env を読み込み: {env_path}")
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            # .env をシェルの環境変数より優先する — プロジェクト単位で上書きするため
                            os.environ[key.strip()] = value.strip()
                return
            current = current.parent


def create_client() -> genai.Client:
    """Vertex AI または APIキー方式で genai.Client を作成"""
    load_env_file()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        print(f"Vertex AI モード: project={project}, location={location}")
        return genai.Client(vertexai=True, project=project, location=location)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        print("APIキーモード")
        return genai.Client(api_key=api_key)
    raise EnvironmentError("GOOGLE_CLOUD_PROJECT または GEMINI_API_KEY を設定してください")


# =============================================================================
# LP用プリセット定義
# =============================================================================

LP_PRESETS = {
    "hero": {
        "name": "ヒーロー画像",
        "aspect_ratio": "16:9",
        "description": "LPのファーストビュー背景。ワイドで没入感のある画像",
        "style_hint": """
写実的で高品質な写真風の画像を生成してください。
- テキストは一切入れないでください — HTMLで重ねるため
- CSS background-image として使うため、中央に重要な被写体を配置しすぎないでください
- 上下左右にグラデーションで暗くなる余白があると、テキストが読みやすくなります
- 明るく、ポジティブで、プロフェッショナルな雰囲気にしてください
""",
    },
    "portrait": {
        "name": "ポートレート",
        "aspect_ratio": "1:1",
        "description": "人物写真。トレーナー紹介やお客様の声用",
        "style_hint": """
人物のポートレート写真風の画像を生成してください。
- テキストは一切入れないでください
- 正方形フォーマット
- 背景はシンプル — 白、ライトグレー、またはぼかした室内
- 人物は胸から上、正面向き、自然な表情
- 清潔感があり、信頼感を与える雰囲気
""",
    },
    "section-bg": {
        "name": "セクション背景",
        "aspect_ratio": "16:9",
        "description": "各セクションの背景画像。抽象的でテキストの邪魔をしない",
        "style_hint": """
Webページのセクション背景として使う抽象的な画像を生成してください。
- テキストは一切入れないでください
- 被写体は控えめに、テクスチャや抽象的なパターン中心
- 全体的に明るめで、上にテキストを重ねても読めるように
- グラデーションや淡い色彩が適切
""",
    },
    "icon": {
        "name": "アイコン・イラスト",
        "aspect_ratio": "1:1",
        "description": "特徴紹介やメリット説明用のアイコン風イラスト",
        "style_hint": """
フラットでモダンなアイコン風のイラストを生成してください。
- テキストは一切入れないでください
- 正方形フォーマット
- シンプルな形状、2-3色のフラットカラー
- 背景は透明感のある薄い色または白
- 概念を一目で伝えるシンボリックなデザイン
""",
    },
    "ogp": {
        "name": "OGP画像",
        "aspect_ratio": "16:9",
        "description": "SNSシェア時に表示されるサムネイル画像",
        "style_hint": """
SNSでシェアされた時にクリックしたくなるサムネイル画像を生成してください。
- テキストは最小限にしてください — タイトルがあれば短く大きく
- 16:9フォーマット
- 視認性が高く、小さいサムネイルでも内容が伝わるように
- 鮮やかで目を引くカラーリング
""",
    },
}

LP_STYLES = {
    "photo-realistic": "写真のようにリアルで高品質な画像。自然な光と影。実在するかのような質感。",
    "warm": "暖かみのある色調。オレンジ〜ベージュ系。親しみやすく、安心感のある雰囲気。",
    "professional": "プロフェッショナルで洗練された雰囲気。クリーンな構図。ビジネス向け。",
    "minimal": "ミニマルで余白を活かしたデザイン。要素を絞り、シンプルに。",
    "bright": "明るく元気な色使い。ポジティブで活動的な印象。フィットネス・健康系に最適。",
}


# =============================================================================
# プロンプト構築
# =============================================================================

def build_prompt(user_prompt: str, preset: Optional[str], style: Optional[str]) -> str:
    """ユーザープロンプトにプリセットとスタイルのヒントを付加"""
    parts = []
    if preset and preset in LP_PRESETS:
        p = LP_PRESETS[preset]
        parts.append(f"【用途】{p['name']} — {p['description']}")
        parts.append(p["style_hint"])
    if style and style in LP_STYLES:
        parts.append(f"【スタイル】{LP_STYLES[style]}")
    parts.append("""
【共通の制約】
- LP（ランディングページ）の素材画像として使います
- HTML/CSSでテキストを重ねるため、画像内にテキストは入れないでください
- 高品質で、Webサイトに載せても違和感のないクオリティにしてください
- 日本のWebサイト向けの画像です
""")
    parts.append(f"【生成する画像の内容】\n{user_prompt}")
    return "\n".join(parts)


# =============================================================================
# 画像生成
# =============================================================================

# 上から順に試す — 高品質優先、失敗したら高速・安価なモデルへフォールバック
IMAGE_GENERATION_MODELS = [
    "gemini-3-pro-image",      # 高品質。GOOGLE_CLOUD_LOCATION=global が必須
    "gemini-2.5-flash-image",  # 高速・安価。global / us-central1 どちらでも動く
]


def generate_with_fallback(client, contents, aspect_ratio: str = "16:9"):
    """フォールバック付きで画像生成"""
    last_error = None
    for model_name in IMAGE_GENERATION_MODELS:
        try:
            print(f"モデル使用: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            )
            return response
        except Exception as e:
            print(f"エラー: {model_name} — {type(e).__name__}")
            last_error = e
    raise Exception(f"すべてのモデルが利用できません。\n最後のエラー: {last_error}")


def generate_lp_image(
    prompt: str,
    output_path: str,
    preset: Optional[str] = None,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    reference: Optional[str] = None,
) -> str:
    """LP用画像を生成して保存"""
    client = create_client()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if aspect_ratio:
        ar = aspect_ratio
    elif preset and preset in LP_PRESETS:
        ar = LP_PRESETS[preset]["aspect_ratio"]
    else:
        ar = "16:9"

    if reference and Path(reference).exists():
        print(f"画像編集モード: {reference}")
        ref_image = Image.open(reference)
        edit_prompt = f"以下の参考画像を元に修正してください。\n\n【修正指示】\n{prompt}\n\n画像内にテキストは入れないでください。"
        response = generate_with_fallback(client, [edit_prompt, ref_image], ar)
    else:
        full_prompt = build_prompt(prompt, preset, style)
        print(f"プリセット: {preset or 'なし'} / スタイル: {style or 'デフォルト'} / アスペクト比: {ar}")
        response = generate_with_fallback(client, full_prompt, ar)

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_data = part.inline_data.data
            image = Image.open(io.BytesIO(image_data))
            image.save(output_path)
            print(f"画像を保存しました: {output_path}")
            return output_path

    raise Exception("画像が生成されませんでした。プロンプトを変えて再試行してください。")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="LP用画像生成スクリプト — image-create.py のLP特化版"
    )
    ap.add_argument("--prompt", "-p", required=True, help="画像の内容を説明するテキスト")
    ap.add_argument("--output", "-o", required=True, help="出力ファイルパス")
    ap.add_argument("--preset", choices=list(LP_PRESETS.keys()),
                    help="LP用プリセット: hero / portrait / section-bg / icon / ogp")
    ap.add_argument("--style", "-s", choices=list(LP_STYLES.keys()),
                    help="スタイル: photo-realistic / warm / professional / minimal / bright")
    ap.add_argument("--aspect-ratio", "-a",
                    choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                    help="アスペクト比 — プリセット指定時は自動設定")
    ap.add_argument("--reference", "-r", help="参考画像のパス — 画像編集モード")
    ap.add_argument("--list-presets", action="store_true", help="プリセット一覧を表示")
    args = ap.parse_args()

    if args.list_presets:
        print("\n=== LP用プリセット一覧 ===\n")
        for key, p in LP_PRESETS.items():
            print(f"  {key:12} {p['aspect_ratio']:5}  {p['name']} — {p['description']}")
        print("\n=== スタイル一覧 ===\n")
        for key, desc in LP_STYLES.items():
            print(f"  {key:16} {desc}")
        return

    generate_lp_image(
        prompt=args.prompt,
        output_path=args.output,
        preset=args.preset,
        style=args.style,
        aspect_ratio=args.aspect_ratio,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
