# Python スタブ生成ツール：要件定義書

## 概要

本ツールは、指定されたディレクトリ配下の Python ファイルに対し、docstring を生成するツールである。

## 設計

### 入力

- ディレクトリパス（例：`./src`）
- このディレクトリ配下にある `.py` ファイルすべてが処理対象となる。
- サブディレクトリも再帰的に含む。
- `.py` ファイル以外は対象外とする。

### 出力

- ユーザーが指定した出力ディレクトリに、入力ディレクトリを複製
- 出力ディレクトリ内の全ての `.py` ファイル（再帰的に処理）に対して、docstring を挿入したファイルを生成

### 処理

- **docstring を挿入**：
  - 関数の定義の下に docstring を挿入。
  - 既にdocstring がある場合は上書きまたはパス
    - これはオプションで指定
    - デフォルトはパス（上書きしない）
  - docstring の内容は LLMを用いて生成
  - LLMの出力は docstring のみとする
  - スタイルは Google, NumPy, reStructuredText スタイル
    - ユーザーが選択可能
    - デフォルトは Google スタイル
    - LLMにプロンプトで「OO-style」と指定する

- **対象関数の範囲**：

  - `def` により定義されたトップレベル関数、およびクラス内のメソッドを対象とする。
  - 内部関数（関数内部で定義された関数）やラムダ式は対象外。
  - `if __name__ == "__main__"` ブロック内の関数も対象外とする。
  - 将来的にはデコレータの扱いについて検討する（現在は未定）。

- **LLM**

  - OpenRouter API
  - OpenAI API (Responses API)
  - ollama（ローカルLLM）

- **LLM 入出力フォーマット**：

  - 入力：関数全体を LLM に次のプロンプトとともに送信する：
    
    ```text
    Generate a [docstring format here]-style docstring for the following Python function. Return only the generated docstring.

    [function code here]
    ```

  - 出力：生成した docstring を出力ファイルに追加。

#### 例

**入力ファイル**

```python
def add(a, b):
    return a + b

class Calculator:
    def subtract(self, x, y):
        return x - y
```

**出力ファイル**

```python
def add(a, b):
    """
    Add two numbers.

    Args:
        a: A number.
        b: A number.

    Returns:
        Sum of a and b.
    """
    return a + b

class Calculator:
    def subtract(self, x, y):
        """
        Subtract one number from another.

        Args:
            x: A number.
            y: A number.

        Returns:
            Result of x - y.
        """
        return x - y
```

### インターフェイス

- コマンドラインツールとして提供

```bash
uv run docstring_generator.py [input_dir] [output_dir] [options]
```

- オプション

  - `--api-key`：OpenRouter, OpenAI API のキー。環境変数（下記参照）を上書きする。文字列
  - `--style`：docstring のスタイル
    - `google`: Google
    - `numpy`: NumPy
    - `rest`: reStructuredText
  - `--overwrite`：既存の docstring を上書きするかどうか（デフォルトはパス）
  - `--llm`：試用する LLM の種類
    - `openai`: OpenAI API
    - `openrouter`: OpenRouter API
    - `ollama`: ollama（ローカルLLM）
      - `--url`：ローカルサーバの URL。文字列
  - `--model`：LLM のモデル名。文字列
  - `--verbose` or `-v`：詳細なログを出力（デフォルトは無効）

- 環境変数

  - `OPENROUTER_API_KEY`：OpenRouter API のキー
  - `OPENAI_API_KEY`：OpenAI API のキー

## 想定ユースケース

- docstring が整備されていないソースコードからドキュメントを作成する手助け。
