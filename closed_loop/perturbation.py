"""LIBERO-PRO perturbation engine.

Adapted from https://github.com/Zxy-MLlab/LIBERO-PRO (MIT License).

Five perturbation types that test different axes of VLA generalization:
  position    (swap)        – swap object start-positions in the BDDL :init block
  object                    – replace objects with visually different variants
  language                  – paraphrase the task instruction
  task                      – change the goal (different language + goal predicate)
  environment               – move the scene to a different table/floor environment

The public entry point is `apply_perturbation(bddl_content, suite, task, kind, seed)`
which returns the modified BDDL string.  For the `language` kind it also returns the
new instruction text so callers can use it without re-parsing the BDDL.
"""
from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import yaml


# ---------------------------------------------------------------------------
# BDDL parser
# ---------------------------------------------------------------------------

class BDDLParser:
    """Extract objects-of-interest and initial On() relations from a BDDL string."""

    def __init__(self, content: str) -> None:
        self.file_content = content
        self.objects_of_interest: List[str] = self._parse_obj_of_interest()
        self.initial_states: Dict[str, str] = self._parse_initial_states()

    def _parse_obj_of_interest(self) -> List[str]:
        m = re.search(r"\(:obj_of_interest(.*?)\)", self.file_content, re.DOTALL)
        if not m:
            return []
        return re.findall(r"(\w+_\d+)", m.group(1))

    def _parse_initial_states(self) -> Dict[str, str]:
        states: Dict[str, str] = {}
        m = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", self.file_content, re.S)
        if not m:
            return states
        for match in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", m.group(1)):
            states[match.group(1)] = match.group(2)
        return states

    @staticmethod
    def extract_language(content: str) -> Optional[str]:
        """Return the raw text inside (:language ...)."""
        m = re.search(r"\(:language\s*(.*?)\)", content, re.S)
        return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Individual perturbators
# ---------------------------------------------------------------------------

class SwapPerturbator:
    """Swap initial On() positions of objects-of-interest (position perturbation)."""

    def __init__(self, parser: BDDLParser, config_path: str) -> None:
        self.parser = parser
        with open(config_path) as f:
            self.config: Dict = yaml.safe_load(f)

    def perturb(self, task_suite_name: str, task_name: str) -> str:
        content = self.parser.file_content
        objs = list(self.parser.objects_of_interest or [])
        if not objs:
            return content

        task_cfg = self.config.get(task_suite_name, {}).get(task_name)
        if task_cfg is None:
            return content

        init = dict(self.parser.initial_states)
        used: set = set()
        random.shuffle(objs)

        def candidates(obj: str) -> List[str]:
            if isinstance(task_cfg, dict):
                pool = task_cfg.get(obj) or task_cfg.get("__any__") or []
                return list(pool)
            return list(task_cfg) if isinstance(task_cfg, list) else []

        for obj in objs:
            if obj in used:
                continue
            pool = [x for x in candidates(obj) if x != obj and x in init and x not in used]
            if not pool:
                continue
            swap = random.choice(pool)
            reg_a, reg_b = init.get(obj), init.get(swap)
            if not reg_a or not reg_b:
                continue
            content, n1 = re.subn(
                rf"\(On\s+{re.escape(obj)}\s+{re.escape(reg_a)}\s*\)",
                f"(On {obj} {reg_b})", content, count=1)
            content, n2 = re.subn(
                rf"\(On\s+{re.escape(swap)}\s+{re.escape(reg_b)}\s*\)",
                f"(On {swap} {reg_a})", content, count=1)
            if n1 and n2:
                init[obj], init[swap] = reg_b, reg_a
                used |= {obj, swap}

        return content


class ObjectReplacePerturbator:
    """Replace task-relevant objects with visually different variants.

    Only substitutions whose target type exists in LIBERO's OBJECTS_DICT are
    applied; variants that require assets from the LIBERO-PRO fork (e.g.
    red_bowl, yellow_stove) are silently skipped so the env always loads
    without KeyError.  When a valid variant is chosen the type annotation in
    the (:objects)/(:fixtures) block is updated alongside every instance-name
    reference, so the environment actually loads the different 3-D mesh.
    """

    # Populated once on first use to avoid importing libero at module load time.
    _VALID_TYPES: Optional[set] = None

    def __init__(self, parser: BDDLParser, config_path: str) -> None:
        self.parser = parser
        with open(config_path) as f:
            self.config: Dict = yaml.safe_load(f)
        if ObjectReplacePerturbator._VALID_TYPES is None:
            try:
                from libero.libero.envs.objects import OBJECTS_DICT
                ObjectReplacePerturbator._VALID_TYPES = set(OBJECTS_DICT.keys())
            except Exception:
                ObjectReplacePerturbator._VALID_TYPES = set()

    @staticmethod
    def _language_span(text: str) -> Optional[Tuple[int, int]]:
        m = re.search(r"\(:language\b.*?\)", text, flags=re.S)
        return m.span() if m else None

    @staticmethod
    def _replace_instance_and_type(content: str, old_type: str, new_type: str) -> str:
        """Replace every occurrence of old_type (both instance names and the
        '- <type>' declaration) with new_type.  Instance numbers are preserved:
        e.g. akita_black_bowl_1 → white_bowl_1."""
        # Replace instance names (with trailing _N suffix)
        content = re.sub(
            re.escape(old_type) + r"(_\d+)",
            new_type + r"\1",
            content,
        )
        # Replace bare type name (covers '- akita_black_bowl' declarations
        # and any remaining unindexed references)
        content = re.sub(
            r"(?<![_\w])" + re.escape(old_type) + r"(?![_\w\d])",
            new_type,
            content,
        )
        return content

    def perturb(self, task_suite_name: str, task_name: str,
                seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        task_cfg: Dict[str, List[str]] = (
            self.config.get(task_suite_name, {}).get(task_name) or {}
        )
        if not task_cfg:
            return self.parser.file_content

        valid = self._VALID_TYPES or set()
        mapping: Dict[str, str] = {}
        for obj_type, cands in task_cfg.items():
            valid_cands = [c for c in (cands or []) if c in valid and c != obj_type]
            if valid_cands:
                mapping[obj_type] = random.choice(valid_cands)

        if not mapping:
            return self.parser.file_content

        content = self.parser.file_content
        # Exclude language block from substitution (instruction text unchanged)
        span = self._language_span(content)
        if span:
            s, e = span
            prefix, lang_block, suffix = content[:s], content[s:e], content[e:]
        else:
            prefix, lang_block, suffix = content, "", ""

        for old_type, new_type in mapping.items():
            prefix = self._replace_instance_and_type(prefix, old_type, new_type)
            suffix = self._replace_instance_and_type(suffix, old_type, new_type)

        return prefix + lang_block + suffix


class LanguagePerturbator:
    """Replace the (:language ...) instruction with a paraphrase."""

    def __init__(self, parser: BDDLParser, config_path: str) -> None:
        self.parser = parser
        with open(config_path, encoding="utf-8") as f:
            self.config: Dict = yaml.safe_load(f) or {}

    def perturb(self, task_suite_name: str, task_name: str,
                seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        candidates = (self.config.get(task_suite_name) or {}).get(task_name) or []
        if not candidates:
            return self.parser.file_content

        new_lang = random.choice(candidates)
        m = re.search(r"\(:language\s*(.*?)\)", self.parser.file_content, re.S)
        if not m:
            return self.parser.file_content

        s, e = m.start(1), m.end(1)
        return self.parser.file_content[:s] + new_lang + self.parser.file_content[e:]


class TaskPerturbator:
    """Replace goal predicate, language, and obj_of_interest simultaneously."""

    def __init__(self, parser: BDDLParser, config_path: str) -> None:
        self.parser = parser
        with open(config_path, encoding="utf-8") as f:
            self.config: Dict = yaml.safe_load(f) or {}

    @staticmethod
    def _outer_block(text: str, head: str) -> Optional[Tuple[int, int]]:
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def perturb(self, task_suite_name: str, task_name: str,
                seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        task_cfg = (self.config.get(task_suite_name) or {}).get(task_name) or {}
        if not task_cfg:
            return self.parser.file_content

        chosen_lang = random.choice(list(task_cfg.keys()))
        chosen = task_cfg[chosen_lang]
        goal = chosen.get("goal")
        objs = chosen.get("obj_of_interest", [])
        if not goal:
            return self.parser.file_content

        content = self.parser.file_content

        # Replace language
        m = re.search(r"\(:language\s*(.*?)\)", content, re.S)
        if m:
            content = content[:m.start(1)] + chosen_lang + content[m.end(1):]

        # Replace goal block
        span = self._outer_block(content, "(:goal")
        if span:
            s, e = span
            content = content[:s] + f"(:goal\n  {goal}\n)" + content[e:]

        # Replace obj_of_interest block
        span = self._outer_block(content, "(:obj_of_interest")
        if span:
            s, e = span
            inner = "\n".join(f"  {o}" for o in objs)
            content = content[:s] + f"(:obj_of_interest\n{inner}\n)" + content[e:]

        return content


class EnvironmentReplacePerturbator:
    """Replace the working environment (table type) in the BDDL."""

    _ALLOWED = {"main_table", "kitchen_table", "living_room_table", "study_table", "floor"}
    _TOKEN = {
        "main_table":        "Tabletop",
        "kitchen_table":     "Kitchen_Tabletop",
        "living_room_table": "Living_Room_Tabletop",
        "study_table":       "Study_Tabletop",
        "floor":             "Floor",
    }
    _FIXTYPE = {
        "main_table":        "table",
        "kitchen_table":     "kitchen_table",
        "living_room_table": "living_room_table",
        "study_table":       "study_table",
        "floor":             "floor",
    }

    def __init__(self, parser: BDDLParser, config_path: str) -> None:
        self.parser = parser
        with open(config_path, encoding="utf-8") as f:
            self.config: Dict = yaml.safe_load(f) or {}

    def _current_env(self, suite: str, task: str) -> Optional[str]:
        entry = (self.config.get(suite) or {}).get(task)
        if isinstance(entry, list):
            return entry[0] if entry else None
        return entry if isinstance(entry, str) else None

    @staticmethod
    def _outer_block(text: str, head: str) -> Optional[Tuple[int, int]]:
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def _build_for(self, cur: str, new_env: str) -> str:
        """Build a BDDL string that replaces `cur` environment with `new_env`."""
        content = self.parser.file_content.replace(cur, new_env)

        # Update problem name token
        content = re.sub(
            r"\(define\s*\(problem\s+LIBERO_[A-Za-z_]*\)",
            f"(define (problem LIBERO_{self._TOKEN[new_env]}_Manipulation)",
            content, count=1,
        )

        # Update fixture type in (:fixtures ...) block
        span = self._outer_block(content, "(:fixtures")
        if span:
            s, e = span
            block = content[s:e]
            block, _ = re.subn(
                rf"(^\s*{re.escape(new_env)}\s*-\s*)([A-Za-z_][A-Za-z0-9_]*)",
                rf"\g<1>{self._FIXTYPE[new_env]}",
                block, count=0, flags=re.M,
            )
            content = content[:s] + block + content[e:]
        return content

    def perturb(self, task_suite_name: str, task_name: str,
                seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        cur = self._current_env(task_suite_name, task_name)
        if not cur or cur not in self._ALLOWED:
            return self.parser.file_content

        candidates = list(self._ALLOWED - {cur})
        new_env = random.choice(candidates)
        return self._build_for(cur, new_env)

    def variants(self, task_suite_name: str, task_name: str) -> List[Tuple[str, str]]:
        """All candidate alternative environments (deterministic, order-fixed).

        Returns a list of (new_env_name, bddl_content) for every allowed
        alternative environment. Empty if the current env is not perturbable.
        Used to average env robustness over ALL environments instead of one
        random pick, which lowers cross-run variance at fixed total episodes.
        """
        cur = self._current_env(task_suite_name, task_name)
        if not cur or cur not in self._ALLOWED:
            return []
        candidates = sorted(self._ALLOWED - {cur})
        return [(new_env, self._build_for(cur, new_env)) for new_env in candidates]


# ---------------------------------------------------------------------------
# Flags + combined perturbator
# ---------------------------------------------------------------------------

@dataclass
class PerturbFlags:
    use_position: bool = False     # swap object positions
    use_object: bool = False       # replace object appearances
    use_language: bool = False     # paraphrase instruction
    use_task: bool = False         # change goal
    use_environment: bool = False  # change environment

    def active_kinds(self) -> List[str]:
        out = []
        if self.use_position:    out.append("position")
        if self.use_object:      out.append("object")
        if self.use_language:    out.append("language")
        if self.use_task:        out.append("task")
        if self.use_environment: out.append("environment")
        return out

    @classmethod
    def from_kind(cls, kind: str) -> "PerturbFlags":
        mapping = {
            "position":    dict(use_position=True),
            "object":      dict(use_object=True),
            "language":    dict(use_language=True),
            "task":        dict(use_task=True),
            "environment": dict(use_environment=True),
            "none":        {},
        }
        if kind not in mapping:
            raise ValueError(f"Unknown perturbation kind {kind!r}; "
                             f"choose from {list(mapping)}")
        return cls(**mapping[kind])


class BDDLCombinedPerturbator:
    """Apply one or more perturbations in sequence to a BDDL string.

    configs dict keys: "position", "object", "language", "task", "environment"
    Values: absolute paths to the corresponding OOD YAML files.
    """

    def __init__(self, configs: Dict[str, str]) -> None:
        self.configs = configs or {}

    @staticmethod
    def _reparse_and_apply(content: str, cls, cfg: str,
                           kwargs: Dict[str, Any],
                           suite: str, task: str) -> str:
        parser = BDDLParser(content)
        p = cls(parser, cfg)
        return p.perturb(task_suite_name=suite, task_name=task, **kwargs)

    def perturb_content(
        self,
        content: str,
        task_suite_name: str,
        task_name: str,
        flags: PerturbFlags,
        seed: Optional[int] = None,
    ) -> str:
        if flags.use_task and (
            flags.use_position or flags.use_object
            or flags.use_language or flags.use_environment
        ):
            raise ValueError("task perturbation is mutually exclusive with all others")

        cur = content

        # position must run first (modifies :init On() relations)
        if flags.use_position:
            cfg = self.configs.get("position")
            if cfg and Path(cfg).exists():
                # SwapPerturbator draws from the global `random` module. Without an
                # explicit seed here Python auto-seeds per process, so the swap (which
                # object lands on the target region) is different every run — making
                # per-task SR uncontrolled and cross-condition comparisons unpaired.
                # Seed deterministically per (seed, task) via a string (sha512-based,
                # PYTHONHASHSEED-independent) so all conditions share the same swap.
                if seed is not None:
                    random.seed(f"pos::{seed}::{task_name}")
                parser = BDDLParser(cur)
                cur = SwapPerturbator(parser, cfg).perturb(task_suite_name, task_name)

        if flags.use_environment:
            cfg = self.configs.get("environment")
            if cfg and Path(cfg).exists():
                cur = self._reparse_and_apply(
                    cur, EnvironmentReplacePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name)

        if flags.use_object:
            cfg = self.configs.get("object")
            if cfg and Path(cfg).exists():
                cur = self._reparse_and_apply(
                    cur, ObjectReplacePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name)

        if flags.use_language:
            cfg = self.configs.get("language")
            if cfg and Path(cfg).exists():
                cur = self._reparse_and_apply(
                    cur, LanguagePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name)

        if flags.use_task:
            cfg = self.configs.get("task")
            if cfg and Path(cfg).exists():
                cur = self._reparse_and_apply(
                    cur, TaskPerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name)

        return cur


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def _default_ood_configs(ood_dir: Optional[str] = None) -> Dict[str, str]:
    """Return absolute paths to the 5 OOD YAML files bundled with this module."""
    if ood_dir is None:
        ood_dir = str(Path(__file__).parent / "libero_ood")
    return {
        "position":    os.path.join(ood_dir, "ood_spatial_relation.yaml"),
        "object":      os.path.join(ood_dir, "ood_object.yaml"),
        "language":    os.path.join(ood_dir, "ood_language.yaml"),
        "task":        os.path.join(ood_dir, "ood_task.yaml"),
        "environment": os.path.join(ood_dir, "ood_environment.yaml"),
    }


def apply_perturbation(
    bddl_content: str,
    suite: str,
    task: str,
    kind: str,
    seed: Optional[int] = None,
    ood_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Apply a single perturbation kind to a BDDL string.

    Returns
    -------
    (new_bddl_content, new_language)
        new_language is the instruction text extracted from the (possibly modified)
        BDDL, or None if the language block is not found.
    """
    if kind == "none":
        lang = BDDLParser.extract_language(bddl_content)
        return bddl_content, lang

    flags = PerturbFlags.from_kind(kind)
    configs = _default_ood_configs(ood_dir)
    perturbator = BDDLCombinedPerturbator(configs)
    new_content = perturbator.perturb_content(
        content=bddl_content,
        task_suite_name=suite,
        task_name=task,
        flags=flags,
        seed=seed,
    )
    new_language = BDDLParser.extract_language(new_content)
    return new_content, new_language


def apply_environment_variants(
    bddl_content: str,
    suite: str,
    task: str,
    ood_dir: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Return (bddl, language) for EVERY candidate alternative environment.

    Unlike apply_perturbation(kind='environment') which picks one random env,
    this enumerates all allowed alternatives so the eval can average robustness
    over environments (lower variance at fixed total episodes). Falls back to a
    single (unchanged) entry if the environment config is missing or the task's
    current environment is not perturbable.
    """
    configs = _default_ood_configs(ood_dir)
    cfg = configs.get("environment")
    fallback = [(bddl_content, BDDLParser.extract_language(bddl_content))]
    if not cfg or not Path(cfg).exists():
        return fallback
    parser = BDDLParser(bddl_content)
    pert = EnvironmentReplacePerturbator(parser, cfg)
    variants = pert.variants(suite, task)
    if not variants:
        return fallback
    return [(content, BDDLParser.extract_language(content)) for _env, content in variants]
