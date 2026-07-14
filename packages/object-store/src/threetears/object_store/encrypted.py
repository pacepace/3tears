"""Streaming client-side encryption wrapping any :class:`ObjectStore`.

:class:`EncryptedObjectStore` composes over another ``ObjectStore`` (S3, filesystem, …) and
transparently encrypts on ``put`` / decrypts on ``open_read`` -- so a large artifact (a DB dump)
never sits whole in memory and never lands on the backend in the clear. Everything else
(``delete`` / ``list_keys`` / ``list_entries`` / ``presigned_get_url``) passes straight through
(listings + deletes operate on opaque keys; a presigned URL yields the *ciphertext*).

**Wire format** (one self-describing stream per object)::

    MAGIC (4)  ||  scrypt salt (16)  ||  frame*  ||  (stream ends after the final frame)
    frame = final-flag (1)  ||  ciphertext-len (4, big-endian)  ||  nonce (12)  ||  AES-256-GCM(ct+tag)

Each frame's AAD is ``pack(">Q?", frame_index, final)`` -- the frame index is authenticated (a
reordered/duplicated frame fails its tag) and the ``final`` flag is authenticated (an attacker who
flips it to truncate the stream trips the tag). A stream that ends before a frame with ``final=1``
is a truncation and raises. The AES key is derived per-object from the passphrase via **scrypt**
(memory-hard -- the passphrase may be human-chosen), salted from the header.
"""

from __future__ import annotations

import os
import struct
from collections.abc import AsyncIterator

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from pydantic import SecretStr

from threetears.media.contracts import ObjectListing, ObjectStore

__all__ = ["EncryptedObjectStore"]

_MAGIC = b"3TB1"  # 3tears backup stream, v1
_SALT_LEN = 16
_NONCE_LEN = 12
_FRAME_HEADER = struct.Struct(">BI")  # final-flag (uint8), ciphertext length (uint32 big-endian)
_AAD = struct.Struct(">Q?")  # frame index (uint64), final flag (bool)
_DEFAULT_FRAME = 1 << 20  # 1 MiB plaintext per frame
#: scrypt cost. 2**18 is the deployment default (memory-hard against a weak passphrase); tests pass a
#: lower factor so a per-object KDF does not dominate a hundred round-trips.
_DEFAULT_SCRYPT_N = 2**18
_SCRYPT_R = 8
_SCRYPT_P = 1
_ENCRYPTED_CONTENT_TYPE = "application/octet-stream"


def _derive_key(passphrase: SecretStr, salt: bytes, *, n: int) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=n, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.get_secret_value().encode("utf-8"))


class _StreamReader:
    """Reads exactly-N bytes across an underlying chunked async byte stream."""

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self._buf = bytearray()
        self._eof = False

    async def read_exact(self, n: int) -> bytes:
        while len(self._buf) < n and not self._eof:
            try:
                self._buf += await anext(self._source)
            except StopAsyncIteration:
                self._eof = True
        if len(self._buf) < n:
            raise ValueError("encrypted stream truncated: fewer bytes than a frame declares")
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


class EncryptedObjectStore:
    """An :class:`ObjectStore` that streams client-side AES-256-GCM over an inner store.

    :param inner: the backend object store the ciphertext is written to / read from.
    :param passphrase: the encryption passphrase (per-object scrypt-derived AES key).
    :param frame_size: plaintext bytes per encrypted frame (memory ceiling per frame).
    :param scrypt_n: scrypt cost factor; keep the default in production.
    """

    def __init__(
        self,
        inner: ObjectStore,
        passphrase: SecretStr,
        *,
        frame_size: int = _DEFAULT_FRAME,
        scrypt_n: int = _DEFAULT_SCRYPT_N,
    ) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self._inner = inner
        self._passphrase = passphrase
        self._frame_size = frame_size
        self._scrypt_n = scrypt_n

    async def _encrypt(self, body: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        salt = os.urandom(_SALT_LEN)
        aesgcm = AESGCM(_derive_key(self._passphrase, salt, n=self._scrypt_n))
        yield _MAGIC + salt

        index = 0
        buf = bytearray()
        async for chunk in body:
            buf += chunk
            while len(buf) >= self._frame_size:
                yield self._frame(aesgcm, index, bytes(buf[: self._frame_size]), final=False)
                del buf[: self._frame_size]
                index += 1
        # the final frame closes the stream — always emitted, even for an empty tail.
        yield self._frame(aesgcm, index, bytes(buf), final=True)

    @staticmethod
    def _frame(aesgcm: AESGCM, index: int, plaintext: bytes, *, final: bool) -> bytes:
        nonce = os.urandom(_NONCE_LEN)
        ciphertext = aesgcm.encrypt(nonce, plaintext, _AAD.pack(index, final))
        return _FRAME_HEADER.pack(int(final), len(ciphertext)) + nonce + ciphertext

    async def _decrypt(self, source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        reader = _StreamReader(source)
        header = await reader.read_exact(len(_MAGIC) + _SALT_LEN)
        if header[: len(_MAGIC)] != _MAGIC:
            raise ValueError("not a 3tears encrypted stream (bad magic)")
        aesgcm = AESGCM(_derive_key(self._passphrase, header[len(_MAGIC) :], n=self._scrypt_n))

        index = 0
        while True:
            final_byte, ciphertext_len = _FRAME_HEADER.unpack(await reader.read_exact(_FRAME_HEADER.size))
            final = bool(final_byte)
            nonce = await reader.read_exact(_NONCE_LEN)
            ciphertext = await reader.read_exact(ciphertext_len)
            plaintext = aesgcm.decrypt(nonce, ciphertext, _AAD.pack(index, final))
            if plaintext:
                yield plaintext
            if final:
                break
            index += 1

    async def put(self, key: str, body: AsyncIterator[bytes], *, content_type: str, size: int | None = None) -> None:
        # size is unknown after framing/encryption, so always stream (multipart) — never a sized single PUT.
        await self._inner.put(key, self._encrypt(body), content_type=_ENCRYPTED_CONTENT_TYPE, size=None)

    def open_read(self, key: str) -> AsyncIterator[bytes]:
        return self._decrypt(self._inner.open_read(key))

    async def delete(self, key: str) -> None:
        await self._inner.delete(key)

    async def delete_many(self, keys: list[str]) -> None:
        await self._inner.delete_many(keys)

    def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:
        return self._inner.list_keys(prefix)

    def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:
        return self._inner.list_entries(prefix)

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        # yields a URL to the CIPHERTEXT — the caller must decrypt client-side.
        return await self._inner.presigned_get_url(key, expires_in=expires_in)
