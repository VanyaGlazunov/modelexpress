# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for _resolve_model_name in modelexpress.metadata.publish."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modelexpress.metadata.publish import _resolve_model_name


def _model_config(model: str, model_weights: str = "") -> SimpleNamespace:
    return SimpleNamespace(model=model, model_weights=model_weights)


class TestResolveModelName:
    def test_uses_model_weights_when_set(self):
        cfg = _model_config(
            model="/root/.cache/vllm/assets/model_streamer/4ef104f5",
            model_weights="s3://bucket/my-model",
        )
        assert _resolve_model_name(cfg) == "s3://bucket/my-model"

    def test_falls_back_to_model_when_weights_empty(self):
        cfg = _model_config(model="org/my-model", model_weights="")
        assert _resolve_model_name(cfg) == "org/my-model"

    def test_falls_back_to_model_when_weights_missing(self):
        cfg = SimpleNamespace(model="org/my-model")
        assert _resolve_model_name(cfg) == "org/my-model"

    def test_env_override_takes_priority_over_model_weights(self):
        cfg = _model_config(
            model="/cache/path",
            model_weights="s3://bucket/my-model",
        )
        with patch.dict("os.environ", {"MX_MODEL_IDENTITY": "custom-identity"}):
            assert _resolve_model_name(cfg) == "custom-identity"

    def test_env_override_takes_priority_over_model(self):
        cfg = _model_config(model="org/my-model")
        with patch.dict("os.environ", {"MX_MODEL_IDENTITY": "custom-identity"}):
            assert _resolve_model_name(cfg) == "custom-identity"

    def test_empty_env_override_is_ignored(self):
        cfg = _model_config(model="org/my-model", model_weights="s3://bucket/my-model")
        with patch.dict("os.environ", {"MX_MODEL_IDENTITY": ""}):
            assert _resolve_model_name(cfg) == "s3://bucket/my-model"

    def test_s3_pod_identity_matches_disk_pod_with_env_override(self):
        """Simulate the disk pod / S3 pod RDMA matching scenario.

        S3 pod: vLLM sets model_weights=s3://... automatically.
        Disk pod: --model is a local path, model_weights is empty,
                  operator sets MX_MODEL_IDENTITY to the S3 URI.
        Both must resolve to the same identity for RDMA to match.
        """
        s3_pod_cfg = _model_config(
            model="/root/.cache/vllm/assets/model_streamer/4ef104f5",
            model_weights="s3://rtc-llm-models/gpt-oss-120b",
        )
        disk_pod_cfg = _model_config(
            model="/root/.cache/vllm/assets/model_streamer/4ef104f5",
            model_weights="",
        )
        with patch.dict("os.environ", {"MX_MODEL_IDENTITY": "s3://rtc-llm-models/gpt-oss-120b"}):
            assert _resolve_model_name(s3_pod_cfg) == _resolve_model_name(disk_pod_cfg)
