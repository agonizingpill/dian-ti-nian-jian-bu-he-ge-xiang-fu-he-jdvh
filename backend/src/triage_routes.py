"""Supervisor-facing API for triage-tree version switching and inspection."""
from __future__ import annotations

import os

import redis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import triage_store
from .auth import current_user, require_roles
from .triage import (
    BUILTIN_SPECS,
    FALLBACK_VERSION,
    TriageSpec,
    SpecLoadError,
    build_model,
    registry,
)
from .utils.audit import log_action

router = APIRouter(prefix="/api/triage", tags=["分流树版本"])

_redis = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"),
                        decode_responses=True)


class ActivateIn(BaseModel):
    version: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=4, max_length=500)


class RegisterIn(BaseModel):
    version: str = Field(pattern=r"^[A-Za-z0-9._-]{2,64}$")
    title: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=500)
    feature_names: list[str] = Field(min_length=1, max_length=12)
    training_x: list[list[float]] = Field(min_length=1, max_length=200)
    training_y: list[str] = Field(min_length=1, max_length=200)
    random_state: int = Field(ge=0, le=10_000_000)
    max_depth: int = Field(ge=1, le=12)


@router.get("/versions")
def versions(user=Depends(current_user)):
    return {"versions": triage_store.list_versions(), "active": triage_store.get_active_pointer()}


@router.get("/active")
def active(user=Depends(current_user)):
    pointer = triage_store.get_active_pointer()
    row = triage_store.get_version_row(pointer["active_version"]) if pointer else None
    return {"active_version": pointer["active_version"] if pointer else FALLBACK_VERSION,
            "pointer": pointer, "version": row}


@router.get("/analysis-jobs")
def analysis_jobs(limit: int = 100, user=Depends(current_user)):
    """Every review sheet with the version actually pinned at its intake."""
    limit = max(1, min(limit, 500))
    return {"jobs": triage_store.list_jobs_with_versions(limit)}


@router.post("/versions/{version}/activate")
def activate(version: str, payload: ActivateIn, user=Depends(require_roles("supervisor"))):
    try:
        pointer = triage_store.activate_version(version, int(user["sub"]), payload.reason)
    except KeyError:
        raise HTTPException(404, "分流树版本不存在")
    except SpecLoadError as exc:
        # A version that fails to build can never become active; the previous
        # pointer stays untouched so the workbench keeps taking orders.
        log_action(int(user["sub"]), "triage_activate_blocked", "triage_version", None,
                   {"version": version, "error": str(exc)})
        raise HTTPException(409, {"code": "TRIAGE_LOAD_FAILED",
                                  "message": f"新版加载失败，已保留当前版本：{exc}",
                                  "active_version": triage_store.get_active_pointer()["active_version"]})
    registry.invalidate(version)
    log_action(int(user["sub"]), "triage_activate", "triage_version", None,
               {"version": version, "reason": payload.reason})
    return pointer


@router.post("/versions/register")
def register(payload: RegisterIn, user=Depends(require_roles("supervisor"))):
    if len(payload.training_x) != len(payload.training_y):
        raise HTTPException(422, "训练特征行数与标签数必须一致")
    width = len(payload.feature_names)
    if any(len(row) != width for row in payload.training_x):
        raise HTTPException(422, "每行训练特征的维度必须与 feature_names 一致")
    spec = TriageSpec(
        version=payload.version,
        title=payload.title,
        description=payload.description or "主管登记的自定义分流树",
        feature_names=tuple(payload.feature_names),
        training_x=tuple(tuple(r) for r in payload.training_x),
        training_y=tuple(payload.training_y),
        params={"max_depth": payload.max_depth, "random_state": payload.random_state},
        built_in=False,
    )
    row = triage_store.register_version(spec)
    registry.invalidate(spec.version)
    log_action(int(user["sub"]), "triage_register", "triage_version", None,
               {"version": spec.version, "load_status": row["load_status"],
                "error": row.get("load_error_message")})
    if row["load_status"] != "ready":
        raise HTTPException(409, {"code": "TRIAGE_LOAD_FAILED",
                                  "message": "版本已登记但模型加载失败，不能启用",
                                  "load_error": row.get("load_error_message")})
    return row


@router.get("/versions/{version}/diff")
def diff(version: str, user=Depends(current_user)):
    """Run both the requested version and the active version over recent intakes."""
    try:
        target_spec = triage_store.load_spec(version)
    except KeyError:
        raise HTTPException(404, "分流树版本不存在")
    pointer = triage_store.get_active_pointer()
    baseline_version = pointer["active_version"] if pointer else FALLBACK_VERSION
    if baseline_version == version:
        baseline_spec = target_spec
    else:
        try:
            baseline_spec = triage_store.load_spec(baseline_version)
        except KeyError:
            baseline_spec = BUILTIN_SPECS[FALLBACK_VERSION]
    target_model = build_model(target_spec)
    baseline_model = build_model(baseline_spec)
    jobs = triage_store.recent_feature_jobs()
    comparison = []
    for job in jobs:
        a = baseline_model.classify(job["features"])
        b = target_model.classify(job["features"])
        comparison.append({
            "analysis_job_id": job["id"], "lift_id": job["lift_id"],
            "features": job["features"],
            "baseline_version": baseline_spec.version,
            "baseline_prediction": a["prediction"],
            "target_version": target_spec.version,
            "target_prediction": b["prediction"],
            "diverges": a["prediction"] != b["prediction"],
        })
    return {"baseline_version": baseline_spec.version, "target_version": version,
            "rows": comparison,
            "divergent_count": sum(1 for r in comparison if r["diverges"])}


@router.post("/analysis-jobs/{job_id}/retry")
def retry_job(job_id: int, user=Depends(require_roles("operator", "supervisor"))):
    try:
        job = triage_store.requeue_job(job_id)
    except KeyError:
        raise HTTPException(404, "复核任务不存在")
    except ValueError:
        raise HTTPException(409, "只有失败的复核任务可以重试，进行中的任务不会重复生成结论")
    _redis.lpush("lift_analysis", str(job_id))
    log_action(int(user["sub"]), "triage_retry", "analysis_job", job_id,
               {"model_version": job["model_version"]})
    return {"job_id": job_id, "status": "queued", "model_version": job["model_version"]}
