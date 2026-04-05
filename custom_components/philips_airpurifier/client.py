"""Client helpers for Philips Air Purifier CoAP communication.

This module wraps the upstream ``philips_airctrl.CoAPClient`` with a small
``PatchedCoAPClient`` subclass that restores two stability fixes present in
the older, battle-tested ``aioairctrl`` library that upstream was forked from.

Scope of the patch (everything else is inherited unchanged):

1. ``_init`` creates the aiocoap ``Context`` with an explicit
   ``transports=["simple6"]`` selection. The upstream library lets aiocoap
   pick its default transport, which can vary per host/network and is a
   known source of flakiness. ``simple6`` is the transport aioairctrl has
   used reliably across dozens of Philips models for years.

2. ``observe_status`` cancels the CoAP observation in a ``finally`` block
   when the caller stops iterating. Without this cancellation the device
   keeps an orphaned observation registration alive; when the integration
   later reconnects, the device may silently refuse to push updates to
   the new observer, which manifests as the "hangs after working for a
   while" symptom the upstream README documents. Restoring the
   ``observation.cancel()`` call is the single biggest stability fix at
   the CoAP layer.

The patched class is re-exported as ``CoAPClient`` so the rest of the
integration (and the test suite) can continue importing the familiar name.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from aiocoap import Context, Message, Unreliable
from aiocoap.numbers.codes import GET
from philips_airctrl import CoAPClient as _UpstreamCoAPClient
from philips_airctrl.coap.encryption import EncryptionContext

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_LOGGER = logging.getLogger(__name__)


class PatchedCoAPClient(_UpstreamCoAPClient):
    """Upstream CoAPClient with two stability fixes from aioairctrl.

    See the module docstring for the rationale. All methods not overridden
    here inherit the upstream behaviour, so future upstream improvements
    flow through automatically.
    """

    async def _init(self) -> None:  # type: ignore[override]
        """Create the aiocoap context with an explicit transport selection."""
        self._client_context = await Context.create_client_context(
            transports=["simple6"],
        )
        self._encryption_context = EncryptionContext()
        try:
            await self._sync()
        except BaseException:
            # Clean up the aiocoap context on any init failure (including
            # CancelledError, which inherits BaseException) to avoid a
            # resource leak when the integration setup is aborted.
            with contextlib.suppress(Exception):
                await self._client_context.shutdown()
            raise

    async def observe_status(self) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
        """Observe status and cancel the CoAP observation on generator exit."""

        def decrypt_status(response: Any) -> dict[str, Any]:
            payload_encrypted = response.payload.decode()
            payload = self._encryption_context.decrypt(payload_encrypted)
            status = json.loads(payload)
            return status["state"]["reported"]  # type: ignore[no-any-return]

        _LOGGER.debug("observing status (patched, with observation cancel)")
        request = Message(
            code=GET,
            transport_tuning=Unreliable,
            uri=f"coap://{self.host}:{self.port}{self.STATUS_PATH}",
        )
        request.opt.observe = 0
        requester = self._client_context.request(request)
        observation = requester.observation
        try:
            response = await requester.response
            yield decrypt_status(response)
            if observation is not None:
                async for response in observation:
                    yield decrypt_status(response)
        finally:
            # Critical: release the device-side observation registration so
            # that a subsequent reconnect gets a fresh push subscription.
            if observation is not None:
                with contextlib.suppress(Exception):
                    observation.cancel()


# Re-export under the familiar name so the rest of the integration (and
# tests that patch ``custom_components.philips_airpurifier.coordinator.CoAPClient``
# etc.) keep working without changes.
CoAPClient = PatchedCoAPClient


async def async_create_client(
    host: str,
    timeout: float = 25,
    create_client: Any | None = None,
) -> PatchedCoAPClient:
    """Create a CoAP client for a host with timeout protection."""
    creator = create_client or PatchedCoAPClient.create
    return await asyncio.wait_for(creator(host), timeout=timeout)


async def async_fetch_status(
    host: str,
    connect_timeout: float = 30,
    status_timeout: float = 30,
    create_client: Any | None = None,
) -> dict[str, Any]:
    """Fetch current status using a temporary CoAP client and shut it down."""
    client = await async_create_client(host, timeout=connect_timeout, create_client=create_client)
    try:
        status, _ = await asyncio.wait_for(client.get_status(), timeout=status_timeout)
        return status
    finally:
        with contextlib.suppress(Exception):
            await client.shutdown()
