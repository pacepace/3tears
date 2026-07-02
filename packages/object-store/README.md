# 3tears-object-store

Streaming S3-compatible object store for large binary artifacts (Path-2 of
the scope-and-objects design): pcaps, DB dumps, rendered reports, evidence.

Implements the dependency-free `ObjectStore` protocol from
`3tears-media-contracts` over any S3-compatible backend (MinIO in dev, S3 in
prod). **Streaming by contract** — uploads move through one part-size buffer
at a time via S3 multipart; downloads yield the response body in chunks — so
a multi-GB object never has to sit whole in a pod's memory.

Keys follow the platform's locked scope-first scheme (`keys.build_object_key`):

```
<customer_id>/<scope>/<category>/<YYYY>/<MM>/<DD>/<object_id>/<filename>
```

Lifted from metallm's `S3Service` and made streaming.

## Dependency note

`aioboto3` (the async S3 client) tracks `aiobotocore`, which caps `botocore`
below the latest sync-`boto3` release. Adding this package therefore pins the
workspace's `botocore`/`boto3` lower and transitively pulls `wrapt` and `lxml`
down a major version. That cap is inherent to using an async S3 client and is
accepted — the full 3tears suite is green under the resolved set. If any
package comes to rely on `wrapt>=2` or `lxml>=6` behavior, add an explicit
lower bound at the workspace level so resolution fails loudly instead of
silently regressing.
