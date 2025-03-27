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

- 新規ディレクトリ（例：`./docstring/src`）
- 新しいディレクトリ `./docstring` を作成し、その中に元ディレクトリと同じ名前（例では `./src`）を配置。
- 例では `./docstring/src` に元ディレクトリの構造を再帰的に複製。
- 各 `.py` ファイルに対して処理を施す。

### 出力ファイルの仕様

- **構造は保持**：関数・クラス・メソッドの名前や引数をコピー。

- **docstring を挿入**：

  - 関数の定義の下に docstring を挿入。
  - 既にdocstring がある場合は上書きする。
  - docstring の内容は LLM（OpenRouter）を用いて生成。
  - LLMの出力は「関数定義 + docstring」のみとする。
  - スタイルは Google スタイルを基本としつつ、生成内容は LLM に任せる。

- **対象関数の範囲**：

  - `def` により定義されたトップレベル関数、およびクラス内のメソッドを対象とする。
  - 内部関数（関数内部で定義された関数）やラムダ式は対象外。
  - `if __name__ == "__main__"` ブロック内の関数も対象外とする。
  - 将来的にはデコレータの扱いについて検討する（現在は未定）。

- **ディレクトリ構造**：

  - 入力ディレクトリ名を `./docstring` 配下にそのままコピーして再構築する。

- **LLM 入出力フォーマット**：

  - 入力：関数全体を LLM に次のプロンプトとともに送信する：
    
    ```text
    Generate a Google-style docstring for the following Python function. Return only the function definition with the generated docstring.

    [function code here]
    ```

  - 出力：対応する関数定義とその docstring を含む形式で出力ファイルに書き込む。
  - docstring の埋め込みは自動で行い、手動修正を許容する設計とする。

#### 例

**入力ファイル（例：`./src/add.py`）**

```python
def add(a, b):
    return a + b

class Calculator:
    def subtract(self, x, y):
        return x - y
```

**出力ファイル（例：`./docstring/src/add.py`）**

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

## 想定ユースケース

- docstring が整備されていないソースコードからドキュメントを作成する手助け。

## 拡張・オプション（将来的に）

- 既存の docstring を維持する or 上書きするオプション。
- ソースコードへの直接的な編集。
