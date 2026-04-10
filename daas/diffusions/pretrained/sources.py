from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ModelSourceType(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL_DIRECTORY = "local_directory"
    SINGLE_FILE = "single_file"


@dataclass(frozen=True)
class ModelSource:
    """
    Describes where a pretrained diffusion pipeline should be loaded from.
    """

    location: str
    source_type: ModelSourceType = ModelSourceType.HUGGINGFACE
    revision: Optional[str] = None
    variant: Optional[str] = None
    subfolder: Optional[str] = None
    use_safetensors: Optional[bool] = None
    token: Optional[str] = None
    local_files_only: bool = False
    extra_load_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_diffusers_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(self.extra_load_kwargs)
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.variant is not None:
            kwargs["variant"] = self.variant
        if self.subfolder is not None:
            kwargs["subfolder"] = self.subfolder
        if self.use_safetensors is not None:
            kwargs["use_safetensors"] = self.use_safetensors
        if self.token is not None:
            kwargs["token"] = self.token
        if self.local_files_only:
            kwargs["local_files_only"] = True
        return kwargs

    @property
    def is_local(self) -> bool:
        return self.source_type in {ModelSourceType.LOCAL_DIRECTORY, ModelSourceType.SINGLE_FILE}


def huggingface_source(
    model_id: str,
    *,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    subfolder: Optional[str] = None,
    use_safetensors: Optional[bool] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
    **extra_load_kwargs: Any,
) -> ModelSource:
    return ModelSource(
        location=model_id,
        source_type=ModelSourceType.HUGGINGFACE,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        use_safetensors=use_safetensors,
        token=token,
        local_files_only=local_files_only,
        extra_load_kwargs=dict(extra_load_kwargs),
    )


def local_directory_source(
    path: str,
    *,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    subfolder: Optional[str] = None,
    use_safetensors: Optional[bool] = None,
    local_files_only: bool = True,
    **extra_load_kwargs: Any,
) -> ModelSource:
    return ModelSource(
        location=path,
        source_type=ModelSourceType.LOCAL_DIRECTORY,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        extra_load_kwargs=dict(extra_load_kwargs),
    )


def single_file_source(
    path: str,
    *,
    token: Optional[str] = None,
    local_files_only: bool = True,
    **extra_load_kwargs: Any,
) -> ModelSource:
    return ModelSource(
        location=path,
        source_type=ModelSourceType.SINGLE_FILE,
        token=token,
        local_files_only=local_files_only,
        extra_load_kwargs=dict(extra_load_kwargs),
    )
