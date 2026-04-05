"""Coordinator for Philips AirPurifier integration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .client import CoAPClient, async_create_client
from .const import (
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_UPDATE_MODE,
    DOMAIN,
    UPDATE_MODE_POLL,
    UPDATE_MODE_PUSH,
    UPDATE_MODE_PUSH_THROTTLED,
)
from .device_models import DEVICE_MODELS
from .model import ApiGeneration, DeviceInformation, DeviceModelConfig

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Stall detection
MISSED_PACKAGE_COUNT = 3
DEFAULT_TIMEOUT = 60
WATCHDOG_CHECK_INTERVAL = 10.0
OBSERVE_ITERATION_TIMEOUT = 90.0
STATUS_FETCH_TIMEOUT = 15.0

# Reconnect backoff
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
RECONNECT_BACKOFF_FACTOR = 2.0

# Poll mode: after this many consecutive failed get_status() calls, the
# coordinator tears down its CoAPClient and builds a fresh one to recover
# from a corrupted underlying aiocoap context.
POLL_MAX_FAILURES_BEFORE_RECREATE = 3


class PhilipsAirPurifierCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage data from Philips AirPurifier via CoAP push."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: CoAPClient,
        host: str,
        device_info: DeviceInformation,
        *,
        update_mode: str = DEFAULT_UPDATE_MODE,
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
        )
        self.client = client
        self.host = host
        self.device_info = device_info

        self._observe_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._timeout: int = DEFAULT_TIMEOUT
        self._watchdog_task: asyncio.Task[None] | None = None
        self._last_update: float = 0.0
        self._device_available = True

        # Update-mode configuration. ``update_mode`` selects between the
        # three strategies documented in const.py: "push", "push_throttled"
        # and "poll". ``update_interval`` is the cadence (seconds) used by
        # "push_throttled" (HA emit rate) and "poll" (network fetch rate).
        # In "push" mode the interval is ignored.
        self._update_mode = update_mode
        self._update_interval = update_interval
        self._latest_status: dict[str, Any] | None = None
        self._throttle_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None

    def _mark_unavailable(self, reason: str) -> None:
        """Mark the device unavailable and log transition once."""
        if self._device_available:
            _LOGGER.warning("Device at %s became unavailable: %s", self.host, reason)
            self._device_available = False

    def _mark_available(self) -> None:
        """Mark the device available and log transition once."""
        if not self._device_available:
            _LOGGER.info("Device at %s is back online", self.host)
        self._device_available = True

    @property
    def model(self) -> str:
        """Return the device model."""
        return self.device_info.model

    @property
    def device_id(self) -> str:
        """Return the device ID."""
        return self.device_info.device_id

    @property
    def device_name(self) -> str:
        """Return the device name."""
        return self.device_info.name

    @property
    def model_config(self) -> DeviceModelConfig:
        """Return the device model configuration."""
        model = self.device_info.model
        model_family = model[:6]
        if model in DEVICE_MODELS:
            return DEVICE_MODELS[model]
        if model_family in DEVICE_MODELS:
            return DEVICE_MODELS[model_family]
        return DeviceModelConfig(api_generation=ApiGeneration.GEN1)

    async def async_set_control_value(self, key: str, value: Any) -> None:
        """Set a single control value on the device."""
        await self.async_set_control_values({key: value})

    async def async_set_control_values(self, values: dict[str, Any]) -> None:
        """Set multiple control values on the device."""
        await self.client.set_control_values(data=values)

    def _emit_observed_status(self, status: dict[str, Any]) -> None:
        """Forward an observed status to HA, honouring the update mode.

        In "push" mode the status is published to HA immediately. In
        "push_throttled" mode the status is cached and the throttle task
        publishes it on a fixed interval. This method is never called in
        "poll" mode because the observe loop is not running.
        """
        self._latest_status = status
        if self._update_mode == UPDATE_MODE_PUSH:
            self.async_set_updated_data(status)

    async def _async_throttle_loop(self) -> None:
        """Publish the latest cached status on a fixed interval."""
        while True:
            await asyncio.sleep(self._update_interval)
            if self._latest_status is not None:
                self.async_set_updated_data(self._latest_status)

    async def _async_poll_loop(self) -> None:
        """Fetch status on a fixed interval instead of observing.

        This is the "poll" update mode. It mirrors the architecture of a
        subprocess-per-poll bridge: each iteration is an independent
        ``get_status()`` request, so stuck-state issues typical of
        long-running observe streams cannot accumulate. After several
        consecutive failures the underlying CoAPClient is torn down and
        rebuilt to recover from corrupted aiocoap state.
        """
        consecutive_failures = 0
        while True:
            try:
                status, timeout = await asyncio.wait_for(
                    self.client.get_status(), timeout=STATUS_FETCH_TIMEOUT
                )
                self._timeout = timeout
                self._last_update = asyncio.get_event_loop().time()
                self._mark_available()
                self._latest_status = status
                self.async_set_updated_data(status)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as err:
                consecutive_failures += 1
                self._mark_unavailable("poll failed")
                _LOGGER.warning(
                    "Poll to %s failed (%d/%d): %s",
                    self.host,
                    consecutive_failures,
                    POLL_MAX_FAILURES_BEFORE_RECREATE,
                    err,
                )
                if consecutive_failures >= POLL_MAX_FAILURES_BEFORE_RECREATE:
                    _LOGGER.warning(
                        "Recreating CoAPClient for %s after %d failures",
                        self.host,
                        consecutive_failures,
                    )
                    with contextlib.suppress(Exception):
                        await self.client.shutdown()
                    try:
                        self.client = await async_create_client(
                            self.host, create_client=CoAPClient.create
                        )
                        consecutive_failures = 0
                    except asyncio.CancelledError:
                        raise
                    except Exception as recreate_err:
                        _LOGGER.warning(
                            "CoAPClient recreation for %s failed: %s",
                            self.host,
                            recreate_err,
                        )
            await asyncio.sleep(self._update_interval)

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from the device (used for initial refresh and fallback)."""
        try:
            status, timeout = await asyncio.wait_for(
                self.client.get_status(), timeout=STATUS_FETCH_TIMEOUT
            )
            self._timeout = timeout
            self._mark_available()
            return status
        except TimeoutError as err:
            self._mark_unavailable("status update timed out")
            msg = f"Timeout communicating with device at {self.host}"
            raise UpdateFailed(msg) from err
        except Exception as err:
            self._mark_unavailable("status update failed")
            msg = f"Error communicating with device at {self.host}"
            raise UpdateFailed(msg) from err

    def _start_observing(self) -> None:
        """Start the update mechanism appropriate for the current mode.

        Historically named _start_observing because push/observe was the
        only mode. It now also spins up the poll loop in "poll" mode.
        """
        # Cancel any existing update tasks so reconfigures on options
        # change replace cleanly.
        for task_attr in ("_observe_task", "_watchdog_task", "_throttle_task", "_poll_task"):
            task = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                setattr(self, task_attr, None)

        if self._update_mode == UPDATE_MODE_POLL:
            self._poll_task = self.hass.async_create_task(
                self._async_poll_loop(),
                f"philips_airpurifier_poll_{self.host}",
            )
            return

        # push or push_throttled → observe + watchdog
        self._observe_task = self.hass.async_create_task(
            self._async_observe_status(),
            f"philips_airpurifier_observe_{self.host}",
        )
        self._watchdog_task = self.hass.async_create_task(
            self._async_watchdog(),
            f"philips_airpurifier_watchdog_{self.host}",
        )

        if self._update_mode == UPDATE_MODE_PUSH_THROTTLED:
            self._throttle_task = self.hass.async_create_task(
                self._async_throttle_loop(),
                f"philips_airpurifier_throttle_{self.host}",
            )

    async def _async_observe_status(self) -> None:
        """Observe device status via CoAP push updates with per-iteration timeout."""
        stream = self.client.observe_status()
        needs_reconnect = False
        try:
            while True:
                try:
                    status = await asyncio.wait_for(
                        stream.__anext__(), timeout=OBSERVE_ITERATION_TIMEOUT
                    )
                except StopAsyncIteration:
                    _LOGGER.debug(
                        "Observation stream completed for %s",
                        self.host,
                    )
                    return
                except TimeoutError:
                    self._mark_unavailable("observe iteration timeout")
                    _LOGGER.warning(
                        "No push update from %s in %.0fs, triggering reconnect",
                        self.host,
                        OBSERVE_ITERATION_TIMEOUT,
                    )
                    needs_reconnect = True
                    break
                self._last_update = asyncio.get_event_loop().time()
                self._mark_available()
                self._emit_observed_status(status)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            self._mark_unavailable("observation stream error")
            _LOGGER.warning(
                "Observation stream error for %s: %s — triggering reconnect",
                self.host,
                err,
            )
            needs_reconnect = True

        if needs_reconnect:
            self.hass.async_create_task(self._async_reconnect())

    async def _async_watchdog(self) -> None:
        """Watch for missed updates and trigger reconnect if needed."""
        while True:
            await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
            if self._last_update <= 0:
                continue
            elapsed = asyncio.get_event_loop().time() - self._last_update
            stall_threshold = self._timeout * MISSED_PACKAGE_COUNT
            if elapsed > stall_threshold:
                self._mark_unavailable("watchdog timeout")
                _LOGGER.warning(
                    "No updates from %s for %.0fs (threshold %.0fs), reconnecting",
                    self.host,
                    elapsed,
                    stall_threshold,
                )
                await self._async_reconnect()

    async def _async_reconnect(self) -> None:
        """Schedule a reconnect if none is in-flight."""
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return

        self._reconnect_task = self.hass.async_create_task(
            self._do_reconnect(),
            f"philips_airpurifier_reconnect_{self.host}",
        )

    async def _do_reconnect(self) -> None:
        """Reconnect with exponential backoff until success or cancellation."""
        self._last_update = 0.0
        delay = RECONNECT_BASE_DELAY
        attempt = 0
        while True:
            attempt += 1
            try:
                with contextlib.suppress(Exception):
                    await self.client.shutdown()

                self.client = await async_create_client(
                    self.host, create_client=CoAPClient.create
                )
                _LOGGER.info(
                    "Reconnected to %s on attempt %d",
                    self.host,
                    attempt,
                )

                try:
                    status, timeout = await asyncio.wait_for(
                        self.client.get_status(), timeout=STATUS_FETCH_TIMEOUT
                    )
                    self._timeout = timeout
                    self._mark_available()
                    self.async_set_updated_data(status)
                except Exception as err:
                    self._mark_unavailable("reconnect status fetch failed")
                    _LOGGER.warning(
                        "Failed to get status after reconnect to %s: %s",
                        self.host,
                        err,
                    )

                self._start_observing()
                return
            except asyncio.CancelledError:
                raise
            except Exception as err:
                _LOGGER.warning(
                    "Reconnect to %s attempt %d failed: %s — retrying in %.1fs",
                    self.host,
                    attempt,
                    err,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * RECONNECT_BACKOFF_FACTOR, RECONNECT_MAX_DELAY)

    async def async_first_refresh_and_observe(self) -> None:
        """Perform first refresh and start observing."""
        try:
            status, timeout = await asyncio.wait_for(
                self.client.get_status(), timeout=STATUS_FETCH_TIMEOUT
            )
            self._timeout = timeout
            self._mark_available()
            self.async_set_updated_data(status)
            _LOGGER.debug("First refresh completed for %s", self.host)
        except Exception as err:
            self._mark_unavailable("initial refresh failed")
            msg = f"Failed to connect to device at {self.host}"
            raise ConfigEntryNotReady(msg) from err

        self._last_update = asyncio.get_event_loop().time()
        self._start_observing()

    async def async_shutdown(self) -> None:
        """Shut down the coordinator."""
        if self._observe_task is not None:
            self._observe_task.cancel()
            self._observe_task = None

        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None

        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._throttle_task is not None:
            self._throttle_task.cancel()
            self._throttle_task = None

        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None

        if self.client is not None:
            with contextlib.suppress(Exception):
                await self.client.shutdown()
