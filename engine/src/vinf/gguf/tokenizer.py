from __future__ import annotations

from dataclasses import dataclass

from vinf.errors import UnsupportedModelError
from vinf.gguf.parser import GGUFFile


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {value: key for key, value in BYTE_ENCODER.items()}


SPECIAL_TOKEN_TYPES = frozenset({3, 4})


@dataclass(frozen=True, slots=True)
class SpecialTokens:
    bos_token_id: int
    eos_token_id: int
    padding_token_id: int | None
    im_start_id: int | None
    im_end_id: int | None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class QwenTokenizer:
    def __init__(
        self,
        *,
        tokens: tuple[str, ...],
        merges: tuple[str, ...],
        token_types: tuple[int, ...],
        special_tokens: SpecialTokens,
        model: str,
        pre: str | None,
        chat_template: str | None,
    ) -> None:
        self.tokens = tokens
        self.merges = merges
        self.token_types = token_types
        self.special_tokens = special_tokens
        self.model = model
        self.pre = pre
        self.chat_template = chat_template
        self._token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self._merge_ranks = {_merge_pair(merge): idx for idx, merge in enumerate(merges)}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}
        # GGUF token types: 3 = control (<|im_start|>), 4 = user-defined (<think>, <tool_call>).
        # Both are matched as atomic special tokens in text, as llama.cpp does; 5 = unused/PAD is not.
        self._special_by_text = {
            token: idx
            for idx, token in enumerate(tokens)
            if idx < len(token_types) and token_types[idx] in SPECIAL_TOKEN_TYPES
        }
        self._special_texts = tuple(sorted(self._special_by_text, key=len, reverse=True))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            special = self._match_special(text, index) if allow_special else None
            if special is not None:
                token, token_id = special
                ids.append(token_id)
                index += len(token)
                continue
            next_special = self._next_special_index(text, index) if allow_special else -1
            end = next_special if next_special >= 0 else len(text)
            segment = text[index:end]
            encoded = "".join(BYTE_ENCODER[byte] for byte in segment.encode("utf-8"))
            for piece in self._bpe(encoded):
                try:
                    ids.append(self._token_to_id[piece])
                except KeyError as exc:
                    raise UnsupportedModelError(f"tokenizer missing BPE piece: {piece!r}") from exc
            index = end
        return ids

    def decode(self, token_ids: list[int] | tuple[int, ...], *, skip_special: bool = False) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.tokens):
                raise UnsupportedModelError(f"token id out of range: {token_id}")
            if skip_special and token_id < len(self.token_types) and self.token_types[token_id] != 1:
                continue
            pieces.append(self.tokens[token_id])
        raw = "".join(pieces)
        out = bytearray()
        passthrough: list[str] = []
        for char in raw:
            byte = BYTE_DECODER.get(char)
            if byte is None:
                if passthrough or char == "<":
                    passthrough.append(char)
                else:
                    passthrough.append(char)
                continue
            if passthrough:
                out.extend("".join(passthrough).encode("utf-8"))
                passthrough.clear()
            out.append(byte)
        if passthrough:
            out.extend("".join(passthrough).encode("utf-8"))
        return out.decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        messages: list[ChatMessage] | list[dict],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        tools: list[dict] | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Qwen3.8 chat template (exact port in vinf.chat_template)."""
        from vinf.chat_template import render_chat

        if not messages:
            raise UnsupportedModelError("no chat messages provided")
        dicts = [
            {"role": m.role, "content": m.content} if isinstance(m, ChatMessage) else m for m in messages
        ]
        return render_chat(
            dicts,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )

    def _match_special(self, text: str, index: int) -> tuple[str, int] | None:
        for token in self._special_texts:
            if text.startswith(token, index):
                return token, self._special_by_text[token]
        return None

    def _next_special_index(self, text: str, index: int) -> int:
        positions = [pos for token in self._special_texts if (pos := text.find(token, index)) >= 0]
        return min(positions) if positions else -1

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token)
        if len(word) <= 1:
            return word
        while True:
            pairs = _pairs(word)
            ranked = [(self._merge_ranks[pair], pair) for pair in pairs if pair in self._merge_ranks]
            if not ranked:
                break
            _, bigram = min(ranked)
            word = _merge_word(word, bigram)
            if len(word) == 1:
                break
        self._bpe_cache[token] = word
        return word


def load_qwen_tokenizer(gguf: GGUFFile) -> QwenTokenizer:
    tokens = _required_list(gguf, "tokenizer.ggml.tokens", str)
    merges = _required_list(gguf, "tokenizer.ggml.merges", str)
    token_types = _required_list(gguf, "tokenizer.ggml.token_type", int)
    if len(tokens) != len(token_types):
        raise UnsupportedModelError("tokenizer token/type length mismatch")
    special = SpecialTokens(
        bos_token_id=_required_int(gguf, "tokenizer.ggml.bos_token_id"),
        eos_token_id=_required_int(gguf, "tokenizer.ggml.eos_token_id"),
        padding_token_id=_optional_int(gguf, "tokenizer.ggml.padding_token_id"),
        im_start_id=_find_token(tokens, "<|im_start|>"),
        im_end_id=_find_token(tokens, "<|im_end|>"),
    )
    return QwenTokenizer(
        tokens=tuple(tokens),
        merges=tuple(merges),
        token_types=tuple(token_types),
        special_tokens=special,
        model=str(gguf.metadata_value("tokenizer.ggml.model")),
        pre=gguf.metadata_value("tokenizer.ggml.pre"),
        chat_template=gguf.metadata_value("tokenizer.chat_template"),
    )


def _merge_pair(merge: str) -> tuple[str, str]:
    parts = merge.split()
    if len(parts) != 2:
        raise UnsupportedModelError(f"invalid tokenizer merge: {merge!r}")
    return parts[0], parts[1]


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


def _merge_word(word: tuple[str, ...], bigram: tuple[str, str]) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    while idx < len(word):
        if idx < len(word) - 1 and (word[idx], word[idx + 1]) == bigram:
            out.append(word[idx] + word[idx + 1])
            idx += 2
        else:
            out.append(word[idx])
            idx += 1
    return tuple(out)


def _required_list(gguf: GGUFFile, key: str, typ: type) -> list:
    value = gguf.metadata_value(key)
    if not isinstance(value, list) or not all(isinstance(item, typ) for item in value):
        raise UnsupportedModelError(f"GGUF tokenizer metadata missing or invalid: {key}")
    return value


def _required_int(gguf: GGUFFile, key: str) -> int:
    value = gguf.metadata_value(key)
    if not isinstance(value, int):
        raise UnsupportedModelError(f"GGUF tokenizer metadata missing or invalid: {key}")
    return value


def _optional_int(gguf: GGUFFile, key: str) -> int | None:
    value = gguf.metadata_value(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise UnsupportedModelError(f"GGUF tokenizer metadata invalid: {key}")
    return value


def _find_token(tokens: list[str], token: str) -> int | None:
    try:
        return tokens.index(token)
    except ValueError:
        return None
