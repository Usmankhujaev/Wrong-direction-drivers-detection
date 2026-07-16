"""Pluggable violation notifiers: webhook, Telegram, MQTT.

Each notifier sends in a daemon thread so a slow or unreachable endpoint never
stalls the video pipeline; failures are printed, not raised.
"""

from __future__ import annotations

import json
import threading
import urllib.request


def _post_json(url, payload, headers=None):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def _in_thread(fn, *args):
    def runner():
        try:
            fn(*args)
        except Exception as exc:  # never let a notifier kill the pipeline
            print(f"notify: {type(exc).__name__}: {exc}")
    threading.Thread(target=runner, daemon=True).start()


class WebhookNotifier:
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers or {}

    def send(self, event: dict):
        _in_thread(_post_json, self.url, event, self.headers)


class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, event: dict):
        text = (f"🚨 Wrong-way vehicle on {event.get('camera', '?')}\n"
                f"track {event.get('track_id')}, lane {event.get('lane')}, "
                f"t={event.get('t_s', 0):.1f}s\n"
                f"snapshot: {event.get('snapshot', '-')}")
        _in_thread(_post_json, self.url, {"chat_id": self.chat_id, "text": text})


class MQTTNotifier:
    def __init__(self, host, port=1883, topic="wrongway/violations"):
        self.host, self.port, self.topic = host, port, topic

    def _publish(self, event):
        import paho.mqtt.publish as publish  # optional dependency
        publish.single(self.topic, json.dumps(event),
                       hostname=self.host, port=self.port)

    def send(self, event: dict):
        _in_thread(self._publish, event)


_KINDS = {"webhook": WebhookNotifier, "telegram": TelegramNotifier,
          "mqtt": MQTTNotifier}


def build_notifiers(configs):
    """Build notifiers from a list of {"kind": ..., **kwargs} dicts."""
    notifiers = []
    for cfg in configs or []:
        cfg = dict(cfg)
        kind = cfg.pop("kind")
        if kind not in _KINDS:
            raise ValueError(f"Unknown notifier kind: {kind!r}")
        notifiers.append(_KINDS[kind](**cfg))
    return notifiers
