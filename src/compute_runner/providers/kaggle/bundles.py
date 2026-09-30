"""Bundles on Kaggle: one private dataset per bundle, named kgr-b-DIGEST40, created once and reused."""

from __future__ import annotations

import json
import os
import time

import requests

from ...store import atomic_json
from .. import RemoteError, classify, http_code, remote_error, short
from ..base import Artifact
from .client import utc

PREFIX = "kgr-b-"
# Failures a diagnostic records beside its row instead of raising.
DIAGNOSTIC_ERRORS = (RemoteError, requests.RequestException, ValueError, KeyError, TypeError)


class BundleDatasets:
    """This account's bundle datasets. Local receipts in uploads/DIGEST/ record each creation."""

    def __init__(self, provider):
        self.provider = provider

    @property
    def api(self):
        return self.provider.api

    @property
    def owner(self) -> str:
        return self.provider.owner

    def ref(self, digest: str) -> str:
        return f"{self.owner}/{PREFIX}{digest[:40]}"

    def _folder(self, digest: str):
        return self.provider.state_dir / "uploads" / digest

    def _accepted_path(self, digest: str):
        # Receipts are account-specific: another account may hold the same bundle.
        return self._folder(digest) / f"accepted-create-{self.owner.lower()}.json"

    # Making a bundle available ----------------------------------------------------------------

    def ensure(self, digest: str) -> str | None:
        """The bundle's pinned dataset version; None while Kaggle processes a creation."""
        ref = self.ref(digest)
        try:
            status = self.api.dataset_status(ref).lower()
        except Exception as error:
            if self._pending_creation(ref, digest, error):
                return None
            return self._create(ref, digest)
        return self._ready(ref, status)

    @staticmethod
    def _ready(ref: str, status: str) -> str | None:
        if status == "ready":
            return ref + "/1"
        if any(word in status for word in ("error", "fail", "deleted")):
            raise RemoteError(f"Dataset {ref}: {status}", "invalid", definitive=True)
        return None

    def _pending_creation(self, ref: str, digest: str, error: Exception) -> bool:
        """Whether a failed status check hides a creation Kaggle already accepted.

        Raises for failures other than the dataset not existing (yet).
        """
        converted = remote_error(error)
        code = http_code(error)
        # Kaggle can accept creation before either status or the owned inventory exposes it.
        # Reuploading then blocks the whole queue and cannot improve visibility.
        if code in {403, 404} and self._accepted_earlier(ref, digest):
            return True
        if converted.kind == "auth":
            if code != 403:
                raise converted from error
            # Ours: Kaggle answers 403 while a new dataset is still processing.
            return self._owned_exists(ref)
        if converted.kind != "missing":
            raise converted from error
        return False

    def _accepted_earlier(self, ref: str, digest: str) -> bool:
        for path in (self._accepted_path(digest), self._folder(digest) / "create-receipt.json"):
            receipt = _read_json(path)
            if (
                receipt.get("ref", "").lower() == ref.lower()
                and str(receipt.get("status", "")).lower() == "ok"
                and not receipt.get("error")
            ):
                return True
        return False

    def _owned_exists(self, ref: str) -> bool:
        # A 403 on GetDatasetStatus can conceal nonexistence. Only a successful
        # authenticated inventory of our own datasets allows creation in that case.
        try:
            return self._owned(ref) is not None
        except Exception as error:
            raise remote_error(error) from error

    def _owned(self, ref: str, *, pages: int | None = None):
        """This account's dataset named ref, from its listing; None if absent from the pages read."""
        page = 1
        while pages is None or page <= pages:
            rows = self.api.dataset_list(mine=True, page=page)
            if not rows:
                return None
            for row in rows:
                if row is not None and (row.ref or "").lower() == ref.lower():
                    return row
            page += 1
        return None

    def _create(self, ref: str, digest: str) -> str | None:
        folder = self._stage(ref, digest)
        try:
            response = self.api.dataset_create_new(
                str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="skip"
            )
            self._save_receipts(ref, digest, response)
            return self._created(ref, response)
        except Exception as error:
            # Deterministic dataset identity is reconciled on the next tick, never versioned here.
            raise remote_error(error) from error

    def _stage(self, ref: str, digest: str):
        folder = self._folder(digest)
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        archive = folder / "payload.zip"
        if not archive.exists():
            os.link(self.provider.state_dir / "bundles" / digest / "payload.zip", archive)
        description = (
            "Private workload snapshot. Original copyright and license terms in the included files "
            "apply; no additional permissions are granted. SHA256: " + digest
        )
        metadata = dict(
            id=ref, title="kgr b " + digest[:40], licenses=[{"name": "other"}], description=description
        )
        atomic_json(folder / "dataset-metadata.json", metadata)
        return folder

    def _save_receipts(self, ref: str, digest: str, response) -> None:
        receipt = {"ref": ref, "created_at": time.time()}
        receipt |= {key: getattr(response, key, None) for key in ("status", "error", "url")}
        atomic_json(self._folder(digest) / "create-receipt.json", receipt)
        if str(receipt["status"]).lower() == "ok" and not receipt["error"]:
            atomic_json(self._accepted_path(digest), receipt)

    def _created(self, ref: str, response) -> str | None:
        """None once Kaggle accepted the creation; it processes the dataset next."""
        if str(getattr(response, "status", "")).lower() == "error" and not response.error:
            raise RemoteError(
                "Kaggle rejected dataset creation without a message", "invalid", definitive=True
            )
        if not response.error:
            return None
        if "already in use by a dataset" in response.error.lower():
            return self._reconcile_conflict(ref)
        raise RemoteError(response.error, classify(response.error), definitive=True)

    def _reconcile_conflict(self, ref: str) -> str | None:
        """The client checks this exact ref again inside create. A newly uploaded dataset can become
        visible between our check and its check; reconcile it instead of blocking every job using it.
        """
        try:
            status = self.api.dataset_status(ref).lower()
        except Exception as error:
            if http_code(error) in {403, 404}:
                raise RemoteError(
                    f"Waiting for dataset {ref} visibility after a create conflict",
                    "transient",
                    definitive=True,
                ) from error
            raise
        return self._ready(ref, status)

    # Diagnostics and cleanup ------------------------------------------------------------------

    def describe(self, alias: str, ref: str, bundle: dict | None) -> dict:
        """What Kaggle and the local receipts say about one input's dataset; never uploads."""
        row = {"alias": alias, "ref": ref}
        if bundle:
            row["bundle_bytes"] = bundle.get("bytes")
            receipt = _read_json(self._folder(bundle["digest"]) / "create-receipt.json")
            if receipt.get("ref", "").lower() == _dataset(ref).lower():
                row["create_receipt"] = receipt
        row |= self._state(_dataset(ref))
        if "status_error" in row:
            row |= self._listing(ref)
        return row

    def _state(self, dataset: str) -> dict:
        found = {}
        for key, form in (("status", None), ("version", "json(current_version_number)")):
            try:
                value = self.api.dataset_status(dataset, format=form)
                found[key] = value if form is None else json.loads(value)
            except DIAGNOSTIC_ERRORS as error:
                found[key + "_error"] = short(error)
        return found

    def _listing(self, ref: str) -> dict:
        """The dataset's row in this account's own listing, read up to ten pages deep."""
        fields = ("ref", "id", "title", "last_updated", "is_private", "total_bytes", "current_version_number")
        try:
            row = self._owned(ref, pages=10)
        except DIAGNOSTIC_ERRORS as error:
            return {"inventory_error": short(error)}
        return {"inventory": {name: short(getattr(row, name, None)) for name in fields}} if row else {}

    def artifacts(self) -> list[Artifact]:
        found, page = [], 1
        while rows := self.api.dataset_list(mine=True, search=PREFIX, page=page):
            for row in rows:
                slug = (row.ref or "").partition("/")[2] if row is not None else ""
                if slug.startswith(PREFIX):
                    updated = row.last_updated
                    modified = utc(updated).timestamp() if updated else None
                    digest = slug.removeprefix(PREFIX)
                    found.append(Artifact("dataset", row.ref, row.total_bytes, modified, digest=digest))
            page += 1
        return found

    def delete(self, ref: str) -> None:
        from kagglesdk.datasets.types.dataset_api_service import ApiDeleteDatasetRequest

        owner, _, slug = ref.partition("/")
        if owner.casefold() != self.owner.casefold() or not slug.startswith(PREFIX):
            raise ValueError(f"Refusing to delete {ref}: not a bundle dataset of this runner")
        request = ApiDeleteDatasetRequest()
        request.owner_slug, request.dataset_slug = owner, slug
        try:
            with self.api.build_kaggle_client() as client:
                response = client.datasets.dataset_api_client.delete_dataset(request)
        except Exception as error:
            raise remote_error(error, mutation=True) from error
        if response.error:
            raise RemoteError(response.error, "invalid", definitive=True)


def _dataset(ref: str) -> str:
    """OWNER/SLUG of a dataset reference that may name a version."""
    return "/".join(ref.split("/")[:2])


def _read_json(path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
