"""Tests for the PatchedCoAPClient subclass."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.philips_airpurifier.client import PatchedCoAPClient


async def test_init_uses_simple6_transport() -> None:
    """_init must create the aiocoap Context with transports=['simple6']."""
    client = PatchedCoAPClient(host="1.2.3.4")

    fake_context = AsyncMock()
    fake_context.shutdown = AsyncMock()

    with (
        patch(
            "custom_components.philips_airpurifier.client.Context.create_client_context",
            new=AsyncMock(return_value=fake_context),
        ) as create_ctx,
        patch(
            "custom_components.philips_airpurifier.client.EncryptionContext",
        ) as enc_cls,
        patch.object(PatchedCoAPClient, "_sync", new=AsyncMock()) as sync_mock,
    ):
        enc_cls.return_value = MagicMock()
        await client._init()

    create_ctx.assert_awaited_once_with(transports=["simple6"])
    sync_mock.assert_awaited_once()
    assert client._client_context is fake_context


async def test_init_shuts_down_context_on_sync_failure() -> None:
    """_init must tear down the aiocoap context if _sync raises."""
    client = PatchedCoAPClient(host="1.2.3.4")

    fake_context = MagicMock()
    fake_context.shutdown = AsyncMock()

    with (
        patch(
            "custom_components.philips_airpurifier.client.Context.create_client_context",
            new=AsyncMock(return_value=fake_context),
        ),
        patch("custom_components.philips_airpurifier.client.EncryptionContext"),
        patch.object(
            PatchedCoAPClient,
            "_sync",
            new=AsyncMock(side_effect=RuntimeError("sync failed")),
        ),
        pytest.raises(RuntimeError, match="sync failed"),
    ):
        await client._init()

    fake_context.shutdown.assert_awaited_once()


async def test_init_shuts_down_context_on_sync_cancellation() -> None:
    """_init must clean up the context even on CancelledError (BaseException)."""
    client = PatchedCoAPClient(host="1.2.3.4")

    fake_context = MagicMock()
    fake_context.shutdown = AsyncMock()

    with (
        patch(
            "custom_components.philips_airpurifier.client.Context.create_client_context",
            new=AsyncMock(return_value=fake_context),
        ),
        patch("custom_components.philips_airpurifier.client.EncryptionContext"),
        patch.object(
            PatchedCoAPClient,
            "_sync",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await client._init()

    fake_context.shutdown.assert_awaited_once()


async def test_init_cleanup_suppresses_shutdown_error() -> None:
    """_init cleanup must not mask the original error if shutdown itself fails."""
    client = PatchedCoAPClient(host="1.2.3.4")

    fake_context = MagicMock()
    fake_context.shutdown = AsyncMock(side_effect=RuntimeError("shutdown boom"))

    with (
        patch(
            "custom_components.philips_airpurifier.client.Context.create_client_context",
            new=AsyncMock(return_value=fake_context),
        ),
        patch("custom_components.philips_airpurifier.client.EncryptionContext"),
        patch.object(
            PatchedCoAPClient,
            "_sync",
            new=AsyncMock(side_effect=RuntimeError("original")),
        ),
        pytest.raises(RuntimeError, match="original"),
    ):
        await client._init()


def _make_response(state: dict) -> MagicMock:
    """Build a fake aiocoap response whose payload decrypts to `state`."""
    payload = json.dumps({"state": {"reported": state}}).encode()
    response = MagicMock()
    response.payload = payload
    return response


def _make_client_with_stub_context(
    *,
    observation_notifications: list[dict] | None,
) -> tuple[PatchedCoAPClient, MagicMock]:
    """Build a PatchedCoAPClient wired to a stub aiocoap context.

    If ``observation_notifications`` is None the requester exposes
    ``observation=None`` (device doesn't support observe); otherwise it
    yields the given dicts and exposes a cancellable observation.
    """
    client = PatchedCoAPClient(host="1.2.3.4")

    # Stub encryption context: payload is already plaintext JSON.
    enc = MagicMock()
    enc.decrypt = lambda encrypted_text: encrypted_text
    client._encryption_context = enc

    initial_state = {"pwr": "1", "pm25": 12}
    initial_response = _make_response(initial_state)

    if observation_notifications is None:
        observation = None
    else:
        observation = MagicMock()

        async def iter_observation():
            for state in observation_notifications:
                yield _make_response(state)

        observation.__aiter__ = lambda self: iter_observation()
        observation.cancel = MagicMock()

    requester = MagicMock()
    requester.response = asyncio.get_event_loop().create_future()
    requester.response.set_result(initial_response)
    requester.observation = observation

    context = MagicMock()
    context.request = MagicMock(return_value=requester)
    client._client_context = context

    return client, observation


async def test_observe_status_yields_initial_then_notifications() -> None:
    """observe_status yields the initial response followed by notifications."""
    client, observation = _make_client_with_stub_context(
        observation_notifications=[{"pwr": "0"}, {"pwr": "1"}],
    )

    results: list[dict] = []
    async for status in client.observe_status():
        results.append(status)

    assert results == [
        {"pwr": "1", "pm25": 12},  # initial
        {"pwr": "0"},
        {"pwr": "1"},
    ]
    assert observation is not None
    observation.cancel.assert_called_once()


async def test_observe_status_cancels_observation_on_aclose() -> None:
    """Observation must be cancelled when the generator is aclose()d.

    This mirrors what happens in production: when HA cancels the observe
    task, Python drives the generator through aclose() which runs the
    finally block and thereby releases the device-side registration.
    """
    client, observation = _make_client_with_stub_context(
        observation_notifications=[{"pwr": "0"}, {"pwr": "1"}, {"pwr": "0"}],
    )

    stream = client.observe_status()
    results: list[dict] = []
    async for status in stream:
        results.append(status)
        if len(results) == 2:
            break
    await stream.aclose()

    assert observation is not None
    observation.cancel.assert_called_once()


async def test_observe_status_without_observation_capability() -> None:
    """If the server doesn't register an observation, yield initial only."""
    client, observation = _make_client_with_stub_context(
        observation_notifications=None,
    )
    assert observation is None

    results: list[dict] = []
    async for status in client.observe_status():
        results.append(status)

    assert results == [{"pwr": "1", "pm25": 12}]


async def test_observe_status_cancel_suppresses_errors() -> None:
    """A failing observation.cancel() must not leak out of the generator."""
    client, observation = _make_client_with_stub_context(
        observation_notifications=[{"pwr": "0"}],
    )
    assert observation is not None
    observation.cancel = MagicMock(side_effect=RuntimeError("cancel failed"))

    # Should not raise despite cancel() blowing up.
    results = [status async for status in client.observe_status()]
    assert len(results) == 2
