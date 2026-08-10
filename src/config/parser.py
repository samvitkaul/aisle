
"""Generic YAML config parser for hierarchical specs"""

from typing import Any

from loguru import logger

from src.utils.common import parse_yaml

INFO    = logger.info
DEBUG   = logger.debug
WARNING = logger.warning
ERROR   = logger.error


def is_external_ref(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    valid_suffixes = ['yaml', 'yml']
    return '@' in value and any(value.endswith(s) for s in valid_suffixes)

def parse_external_ref(ref_str: Any) -> tuple[str, str]:
    if '@' not in ref_str:
        raise ValueError(f'Invalid reference format: {ref_str}')

    parts = ref_str.split('@')
    if len(parts) != 2:
        raise ValueError(f'Invalid reference format: {ref_str} (multiple "@" found)')

    name, filepath = parts
    return name, filepath

def _resolve_refs_recursive(
        config_dict: dict[str, Any],
        loaded_cache: dict[str, Any] | None = None,
        visited_refs: set[str] | None = None,
        ) -> dict[str, Any]:

    if loaded_cache is None:
        loaded_cache = {}

    if visited_refs is None:
        visited_refs = set()


    resolved = {}

    for key, value in config_dict.items():
        if is_external_ref(value):
            DEBUG(f'parsing external ref under {key}')
            ref_name, ref_path = parse_external_ref(value)

            #Check for cycles
            if ref_path in visited_refs:
                ERROR(f'Circular reference detected: {ref_path}')
                raise ValueError(f'Circular reference: {ref_path}')

            #Load the referenced YAML file (using cache)
            if ref_path not in loaded_cache:
                DEBUG(f'Loading external ref {ref_path}')
                loaded_cache[ref_path] = parse_yaml(ref_path)

            referenced_content = loaded_cache[ref_path]

            #Extract the specific named item from the referenced file
            if ref_name not in referenced_content:
                ERROR(f'Reference {ref_name} not found in {ref_path}')
                raise KeyError(
                        f'Reference {ref_name} not found in {ref_path}'
                        f'Available: {list(referenced_content.keys())}'
                        )

            extracted_value = referenced_content[ref_name]

            #Recursively resolve references in the extracted value
            visited_refs.add(ref_path)
            if isinstance(extracted_value, dict):
                extracted_value = _resolve_refs_recursive(
                        extracted_value,
                        loaded_cache=loaded_cache,
                        visited_refs=visited_refs.copy(),
                        )
            visited_refs.discard(ref_path)

            resolved[key] = extracted_value

        elif isinstance(value, dict):
            #Recursively process nested dicts
            DEBUG(f'Recursively process nested dicts under {key}')
            resolved[key] = _resolve_refs_recursive(
                    value,
                    loaded_cache=loaded_cache,
                    visited_refs=visited_refs,
                    )

        elif isinstance(value, list):
            raise TypeError(f'Lists are not allowed. Found under key: {key}')

        else:
            #primitive value, keep as is
            resolved[key] = value

    return resolved

def _is_numeric_leaf_map(d: dict[str, Any]) -> bool:
    """
       A child dict is a <leaf param map> iff it is non-empty
       and every value is a primitive number (int, float) not bool.

       Catches throughput / count tables that look like
       ``{fp8: 1024, bf16: 512}``

       These dicts are *data* not model entities, so they must have
       a name field stamped into them

       Real Pydantic model dicts virtually always include atleast one
       non-numeric value (say KnobVal) - so this test reliably distinguishes them
    """

    if not d:
        return False

    for v in d.values():
        if isinstance(v, bool):
            return False
        if not isinstance(v, (int, float)):
            return False

    return True

def _inject_names_recursive(
        config_dict: dict[str, Any],
        parent_key: str | None = None,
        ignore_keys: list[str] | None = None
        ) -> dict[str, Any]:

    """
       Recursively inject ``name: <child_key>`` into each nested dict

       compared to naive walk, this skips injection when the child dict
       is a *numeric leaf map*

       Other guards:
         - root dict itself is never renamed (no parent)
         - explicitly-set ``name`` fields are preserved
         - ``ignore_keys`` allows callers to opt specific parent keys out
           of injection (kept for backwards compatibility)
    """
    result : dict[str, Any] = {}

    #first process all children
    for key, value in config_dict.items():
        if isinstance(value, dict):
            result[key] = _inject_names_recursive(
                    value,
                    parent_key=key,
                    ignore_keys=ignore_keys,
                    )
        else:
            result[key] = value

    #then optionally inject name for *this* dict
    if parent_key is None or 'name' in result:
        return result
    if ignore_keys and parent_key in ignore_keys:
        return result
    if _is_numeric_leaf_map(result):
        return result
    return {'name': parent_key, **result}

def _inject_names(config_dict: dict[str, Any], ignore_keys: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            result[key] = _inject_names_recursive(value, parent_key=key, ignore_keys=ignore_keys)
        else:
            result[key] = value
    return result

def _get_nested_value(config_dict: dict[str, Any], path: str) -> Any:
    """
       Get a value from nested dict using dot separated path
    """
    if not path.startswith('.'):
        raise ValueError(f'override path must start with .: {path}')

    #remove leading '.' and split by '.'
    path_parts = path[1:].split('.')

    current = config_dict
    for i, part in enumerate(path_parts):
        if not isinstance(current, dict):
            raise KeyError(
                    f'Cannot navigate path {path}'
                    f'expected dict at level {".".join(path_parts[:i])}'
                    f'got {type(current).__name__}'
                    )

        if part not in current:
            raise KeyError(
                    f'Path {path} not found: key {part} not found at level {".".join(path_parts[:i])}'
                    f'Available: {list(current.keys())}'
                    )

        current = current[part]

    return current

def _set_nested_value(config_dict: dict[str, Any], path: str, value: Any) -> Any:
    """
       Set a value from nested dict using dot separated path
    """
    if not path.startswith('.'):
        raise ValueError(f'override path must start with .: {path}')

    #remove leading '.' and split by '.'
    path_parts = path[1:].split('.')

    if not path_parts:
        raise ValueError(f'invalid override path: {path}')

    current = config_dict
    for i, part in enumerate(path_parts[:-1]):
        if not isinstance(current, dict):
            raise KeyError(
                    f'Cannot navigate path {path}'
                    f'expected dict at level {".".join(path_parts[:i])}'
                    f'got {type(current).__name__}'
                    )

        if part not in current:
            raise KeyError(
                    f'Path {path} not found: key {part} not found at level {".".join(path_parts[:i])}'
                    f'Available: {list(current.keys())}'
                    )

        current = current[part]

    #Set the final value
    final_key = path_parts[-1]
    if not isinstance(current, dict):
        raise KeyError(
                f'Cannot set path {path}'
                f'expected dict at parent level, got {type(current).__name__}'
                )

    if final_key not in current:
        raise KeyError(
                f'Path {path} not found: key {final_key} not found at final level'
                f'Available: {list(current.keys())}'
                )

    DEBUG(f'Override: {final_key} = {value}')
    current[final_key] = value
    return


def _apply_overrides(config_dict: dict[str, Any], overrides: dict[str, Any]) -> None:
    for path, value in overrides.items():
        _set_nested_value(config_dict, path, value)


def parse_config_with_refs(
        config_file: str,
        inject_names: bool = False,
        ignore_keys: list[str] | None = None,
        overrides: dict[str, Any] | None = None,
        ) -> dict[str, Any]:

    """
       Parse YAML config and resolve all external refs
    """

    #Load the main cfg file
    main_config = parse_yaml(config_file)

    if not isinstance(main_config, dict):
        raise TypeError(
                f"Config {config_file} must contain a YAML dict/mapping"
                f"got {type(main_config).__name__}"
                )

    resolved_config = _resolve_refs_recursive(main_config)

    if inject_names:
        resolved_config = _inject_names(resolved_config, ignore_keys)

    if overrides:
        _apply_overrides(resolved_config, overrides)

    return resolved_config

